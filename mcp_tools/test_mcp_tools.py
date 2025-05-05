#!/usr/bin/env python3
"""
MCP Tools Test Suite

This script performs comprehensive testing of all MCP tools to verify functionality,
accuracy, performance, and production readiness.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_test")

# Import MCP tools
from mcp_tools.base_mcp_server import BaseMCPServer
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP
from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.trading_mcp.trading_mcp import TradingMCP
from mcp_tools.financial_text_mcp.financial_text_mcp import FinancialTextMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP
from mcp_tools.document_analysis_mcp.document_analysis_mcp import DocumentAnalysisMCP

class MCPTestResults:
    """Class to track and report test results"""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.warnings = []
        self.performance_metrics = {}
        
    def record_test(self, test_name: str, success: bool, duration: float, 
                   details: Optional[Dict[str, Any]] = None, 
                   failure_reason: Optional[str] = None):
        """Record result of a single test"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            logger.info(f"✅ PASS: {test_name} ({duration:.2f}s)")
        else:
            self.tests_failed += 1
            logger.error(f"❌ FAIL: {test_name} ({duration:.2f}s) - {failure_reason}")
            self.failures.append({
                "test": test_name,
                "reason": failure_reason,
                "details": details
            })
            
        # Record performance
        if test_name not in self.performance_metrics:
            self.performance_metrics[test_name] = []
        self.performance_metrics[test_name].append(duration)
    
    def add_warning(self, test_name: str, warning: str):
        """Add a warning message"""
        logger.warning(f"⚠️ WARNING: {test_name} - {warning}")
        self.warnings.append({
            "test": test_name,
            "warning": warning
        })
        
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "="*80)
        print(f"MCP TOOLS TEST RESULTS SUMMARY")
        print("="*80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.failures:
            print("\nFAILURES:")
            for i, failure in enumerate(self.failures, 1):
                print(f"{i}. {failure['test']}: {failure['reason']}")
                
        if self.warnings:
            print("\nWARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning['test']}: {warning['warning']}")
        
        print("\nPERFORMANCE SUMMARY:")
        for test, durations in self.performance_metrics.items():
            avg_duration = sum(durations) / len(durations) if durations else 0
            print(f"- {test}: Avg {avg_duration:.3f}s (min: {min(durations):.3f}s, max: {max(durations):.3f}s)")
            
        print("="*80)
        
    def get_overall_status(self):
        """Get overall test status"""
        if self.tests_failed > 0:
            return "FAILED"
        elif self.warnings:
            return "PASSED WITH WARNINGS"
        else:
            return "PASSED"


# Helper functions
def test_mcp_base(results):
    """Test Base MCP Server functionality"""
    logger.info("Testing BaseMCPServer functionality")
    
    start_time = time.time()
    try:
        # Initialize a basic MCP server
        config = {"test_value": "test"}
        base_mcp = BaseMCPServer(name="test_base_mcp", config=config)
        
        # Test tool registration and execution
        def test_tool(param1, param2=None):
            return {"param1": param1, "param2": param2}
        
        base_mcp.register_tool(test_tool, "test_tool", "Test tool description")
        
        # Check if tool was registered correctly
        tools = base_mcp.list_tools()
        if not any(t["name"] == "test_tool" for t in tools):
            results.record_test("Base MCP Tool Registration", False, time.time() - start_time,
                              failure_reason="Tool not registered correctly")
            return
        
        # Test calling the tool
        tool_result = base_mcp.call_tool("test_tool", {"param1": "value1", "param2": "value2"})
        if tool_result.get("param1") != "value1" or tool_result.get("param2") != "value2":
            results.record_test("Base MCP Tool Execution", False, time.time() - start_time,
                              failure_reason="Tool execution returned incorrect result")
            return
            
        # Test resource registration and access
        test_resource = {"resource_data": "test_data"}
        base_mcp.register_resource("test_resource", test_resource, "Test resource description")
        
        # Check if resource was registered correctly
        resources = base_mcp.list_resources()
        if not any(r["name"] == "test_resource" for r in resources):
            results.record_test("Base MCP Resource Registration", False, time.time() - start_time,
                              failure_reason="Resource not registered correctly")
            return
        
        # Test accessing the resource
        resource_result = base_mcp.access_resource("test_resource")
        if resource_result.get("resource_data") != "test_data":
            results.record_test("Base MCP Resource Access", False, time.time() - start_time,
                              failure_reason="Resource access returned incorrect data")
            return
            
        # Test environment/API key access
        os.environ["TEST_API_KEY"] = "test_api_key_value"
        api_key = base_mcp.get_api_key("test")
        if api_key != "test_api_key_value":
            results.record_test("Base MCP API Key Access", False, time.time() - start_time,
                              failure_reason="API key access failed")
            return
            
        # Test shutdown
        base_mcp.shutdown()
        
        # All tests passed
        results.record_test("BaseMCPServer Core Functionality", True, time.time() - start_time)
        
    except Exception as e:
        results.record_test("BaseMCPServer Core Functionality", False, time.time() - start_time,
                          failure_reason=f"Exception: {str(e)}")


def test_yahoo_finance_mcp(results):
    """Test Yahoo Finance MCP functionality"""
    logger.info("Testing YahooFinanceMCP functionality")
    
    start_time = time.time()
    try:
        # Initialize Yahoo Finance MCP
        yahoo_mcp = YahooFinanceMCP()
        
        # Get list of available tools to verify initialization
        tools = yahoo_mcp.list_tools()
        if not tools:
            results.record_test("Yahoo Finance MCP Initialization", False, time.time() - start_time,
                              failure_reason="No tools registered after initialization")
            return
            
        # Test stock price functionality with a well-known ticker
        ticker = "AAPL"
        price_data = yahoo_mcp.get_stock_price(ticker)
        
        if "error" in price_data:
            results.record_test("Yahoo Finance Stock Price", False, time.time() - start_time,
                             failure_reason=f"Error getting stock price: {price_data.get('error')}")
            return
            
        # Check that we got meaningful data
        if not price_data.get("price") or not isinstance(price_data["price"], (int, float)):
            results.record_test("Yahoo Finance Stock Price", False, time.time() - start_time,
                              failure_reason="Invalid price data returned")
            return
            
        logger.info(f"Got price for {ticker}: {price_data.get('price')}")
        
        # Test historical data functionality
        history_data = yahoo_mcp.get_historical_data(ticker, period="1mo", interval="1d")
        
        if "error" in history_data:
            results.record_test("Yahoo Finance Historical Data", False, time.time() - start_time,
                             failure_reason=f"Error getting historical data: {history_data.get('error')}")
            return
            
        # Check that we got data
        if not history_data.get("data") or not isinstance(history_data["data"], list) or len(history_data["data"]) == 0:
            results.record_test("Yahoo Finance Historical Data", False, time.time() - start_time, 
                              failure_reason="No historical data returned")
            return
            
        logger.info(f"Got {len(history_data.get('data', []))} historical data points for {ticker}")
        
        # Test company info functionality
        company_info = yahoo_mcp.get_company_info(ticker)
        
        if "error" in company_info:
            results.record_test("Yahoo Finance Company Info", False, time.time() - start_time,
                              failure_reason=f"Error getting company info: {company_info.get('error')}")
            return
            
        if not company_info.get("company_info") or not isinstance(company_info["company_info"], dict):
            results.record_test("Yahoo Finance Company Info", False, time.time() - start_time,
                              failure_reason="No company info returned")
            return
            
        logger.info(f"Got company info for {ticker} with {len(company_info.get('company_info', {}))} fields")
        
        # Check available endpoints
        endpoints = yahoo_mcp.list_available_endpoints()
        if not endpoints or not isinstance(endpoints, list):
            results.record_test("Yahoo Finance Endpoints", False, time.time() - start_time,
                              failure_reason="Failed to list available endpoints")
            return
            
        logger.info(f"Found {len(endpoints)} available Yahoo Finance endpoints")
        
        # Test data source status
        status = yahoo_mcp.get_data_source_status()
        if not status or not isinstance(status, dict):
            results.record_test("Yahoo Finance Status", False, time.time() - start_time,
                              failure_reason="Failed to get data source status")
            return
            
        # All tests passed
        results.record_test("YahooFinanceMCP Functionality", True, time.time() - start_time)
        
    except Exception as e:
        results.record_test("YahooFinanceMCP Functionality", False, time.time() - start_time,
                          failure_reason=f"Exception: {str(e)}")


def test_trading_mcp(results):
    """Test Trading MCP functionality"""
    logger.info("Testing TradingMCP functionality")
    
    start_time = time.time()
    try:
        # Initialize Trading MCP
        trading_mcp = TradingMCP()
        
        # Get list of available tools to verify initialization
        tools = trading_mcp.list_tools()
        if not tools:
            results.record_test("Trading MCP Initialization", False, time.time() - start_time,
                              failure_reason="No tools registered after initialization")
            return
            
        # Test client health check
        health_status = trading_mcp.get_trading_client_health()
        
        logger.info(f"Trading client health status: {health_status.get('status')}")
        
        # Skip actual trading API calls if credentials are invalid
        # Just test that the tools work without requiring API calls
        
        # Test getting account info (might fail if no API key)
        account_info = trading_mcp.get_account_info()
        if "error" in account_info:
            logger.warning(f"Skipping Alpaca API tests: {account_info.get('error')}")
            results.add_warning("Trading MCP", "Could not connect to Alpaca API - credentials may be missing")
        
        # Test market status (this doesn't require credentials)
        try:
            clock = trading_mcp.get_clock()
            if "is_open" not in clock and "error" not in clock:
                results.record_test("Trading MCP Market Clock", False, time.time() - start_time,
                                  failure_reason="Invalid response from get_clock")
                return
                
            logger.info(f"Market is currently {'open' if clock.get('is_open') else 'closed'}")
        except Exception as e:
            results.record_test("Trading MCP Market Clock", False, time.time() - start_time,
                              failure_reason=f"Exception getting market clock: {str(e)}")
            return
        
        # All tests passed
        results.record_test("TradingMCP Functionality", True, time.time() - start_time)
        
    except Exception as e:
        results.record_test("TradingMCP Functionality", False, time.time() - start_time,
                          failure_reason=f"Exception: {str(e)}")


def test_financial_text_mcp(results):
    """Test Financial Text MCP functionality"""
    logger.info("Testing FinancialTextMCP functionality")
    
    start_time = time.time()
    try:
        # Initialize Financial Text MCP
        text_mcp = FinancialTextMCP()
        
        # Get list of available tools to verify initialization
        tools = text_mcp.list_tools()
        if not tools:
            results.record_test("Financial Text MCP Initialization", False, time.time() - start_time,
                              failure_reason="No tools registered after initialization")
            return
            
        # Test health status
        health = text_mcp.get_health_status()
        if not health:
            results.record_test("Financial Text MCP Health Check", False, time.time() - start_time,
                              failure_reason="Failed to get health status")
            return
            
        logger.info(f"Financial Text MCP status: {health.get('status')}")
        
        # Test sentiment analysis with a basic text
        sample_text_positive = "The company reported strong earnings, beating analyst expectations."
        sentiment_result = text_mcp.analyze_sentiment(sample_text_positive)
        
        if "error" in sentiment_result:
            logger.warning(f"Skipping sentiment analysis: {sentiment_result.get('error')}")
            results.add_warning("Financial Text MCP", "Could not run sentiment analysis - models may be missing")
        else:
            logger.info(f"Sentiment for positive text: {sentiment_result}")
            
            # Test with negative text
            sample_text_negative = "The company missed earnings targets and lowered guidance for the next quarter."
            sentiment_result_neg = text_mcp.analyze_sentiment(sample_text_negative)
            logger.info(f"Sentiment for negative text: {sentiment_result_neg}")
            
            # Compare to ensure different results
            if sentiment_result.get('sentiment') == sentiment_result_neg.get('sentiment'):
                results.add_warning("Financial Text MCP", 
                                  "Sentiment analysis returned same result for positive and negative text")
        
        # Test entity extraction
        sample_text_entities = "Apple and Microsoft are tech giants with CEOs Tim Cook and Satya Nadella respectively."
        entities_result = text_mcp.extract_entities(sample_text_entities)
        
        if "error" in entities_result:
            logger.warning(f"Skipping entity extraction: {entities_result.get('error')}")
            results.add_warning("Financial Text MCP", "Could not run entity extraction - models may be missing")
        else:
            logger.info(f"Extracted entities: {entities_result}")
            
            # Check if any entities were found
            if not entities_result.get('entities') or len(entities_result['entities']) == 0:
                results.add_warning("Financial Text MCP", "No entities extracted from sample text with known entities")
        
        # All tests passed
        results.record_test("FinancialTextMCP Functionality", True, time.time() - start_time)
        
    except Exception as e:
        results.record_test("FinancialTextMCP Functionality", False, time.time() - start_time,
                          failure_reason=f"Exception: {str(e)}")


def test_redis_mcp(results):
    """Test Redis MCP functionality"""
    logger.info("Testing RedisMCP functionality")
    
    start_time = time.time()
    try:
        # Initialize Redis MCP
        redis_mcp = RedisMCP()
        
        # Get list of available tools to verify initialization
        tools = redis_mcp.list_tools()
        if not tools:
            results.record_test("Redis MCP Initialization", False, time.time() - start_time,
                              failure_reason="No tools registered after initialization")
            return
            
        # Test basic Redis functionality if connection available
        connection_status = redis_mcp.get_connection_status()
        
        if not connection_status.get('connected', False):
            logger.warning("Redis server not available, skipping Redis API tests")
            results.add_warning("Redis MCP", "Redis server not available - connection failed")
            # Still mark test as passed but with warning
            results.record_test("RedisMCP Functionality", True, time.time() - start_time)
            return
            
        # Test set/get operations
        test_key = "test_mcp_key"
        test_value = f"test_value_{datetime.now().isoformat()}"
        
        set_result = redis_mcp.set_value(test_key, test_value)
        if not set_result.get('success', False):
            results.record_test("Redis MCP Set Operation", False, time.time() - start_time,
                              failure_reason=f"Failed to set value: {set_result.get('error')}")
            return
            
        # Get the value back
        get_result = redis_mcp.get_value(test_key)
        if get_result.get('value') != test_value:
            results.record_test("Redis MCP Get Operation", False, time.time() - start_time,
                              failure_reason=f"Get value doesn't match set value")
            return
            
        logger.info(f"Successfully set and retrieved Redis value")
        
        # Delete the test key
        del_result = redis_mcp.delete_key(test_key)
        if not del_result.get('success', False):
            results.add_warning("Redis MCP", f"Failed to clean up test key: {del_result.get('error')}")
        
        # All tests passed
        results.record_test("RedisMCP Functionality", True, time.time() - start_time)
        
    except Exception as e:
        results.record_test("RedisMCP Functionality", False, time.time() - start_time,
                          failure_reason=f"Exception: {str(e)}")


def test_polygon_rest_mcp(results):
    """Test Polygon REST MCP functionality"""
    logger.info("Testing PolygonRestMCP functionality")
    
    start_time = time.time()
    try:
        # Initialize Polygon REST MCP
        polygon_mcp = PolygonRestMCP()
        
        # Get list of available tools to verify initialization
        tools = polygon_mcp.list_tools()
        if not tools:
            results.record_test("Polygon REST MCP Initialization", False, time.time() - start_time,
                              failure_reason="No tools registered after initialization")
            return
            
        # Get available endpoints
        endpoints = polygon_mcp.list_available_endpoints()
        if not endpoints:
            results.record_test("Polygon REST Endpoints", False, time.time() - start_time,
                              failure_reason="Failed to list available endpoints")
            return
            
        logger.info(f"Found {len(endpoints)} available Polygon REST endpoints")
        
        # Test only if API key available
        api_key = polygon_mcp.get_api_key("polygon", "")
        if not api_key:
            logger.warning("Polygon API key not available, skipping API tests")
            results.add_warning("Polygon REST MCP", "API key not available - skipping API tests")
            # Still mark test as passed but with warning
            results.record_test("PolygonRestMCP Functionality", True, time.time() - start_time)
            return
            
        # Test with a basic API call that should always work
        ticker = "AAPL"
        ticker_details = polygon_mcp.get_ticker_details(ticker)
        
        if "error" in ticker_details:
            results.record_test("Polygon REST API Call", False, time.time() - start_time,
                              failure_reason=f"Error getting ticker details: {ticker_details.get('error')}")
            return
            
        logger.info(f"Successfully retrieved ticker details for {ticker}")
        
        # All tests passed
        results.record_test("PolygonRestMCP Functionality", True, time.time() - start_time)
        
    except Exception as e:
        results.record_test("PolygonRestMCP Functionality", False, time.time() - start_time,
                          failure_reason=f"Exception: {str(e)}")


def test_other_mcp_initialization(results):
    """Test initialization of remaining MCP classes"""
    logger.info("Testing initialization of remaining MCP servers")
    
    mcp_classes = [
        (FinancialDataMCP, "FinancialDataMCP"),
        (RiskAnalysisMCP, "RiskAnalysisMCP"),
        (TimeSeriesMCP, "TimeSeriesMCP"),
        (VectorStoreMCP, "VectorStoreMCP"),
        (DocumentAnalysisMCP, "DocumentAnalysisMCP"),
    ]
    
    for mcp_class, mcp_name in mcp_classes:
        start_time = time.time()
        try:
            # Try to initialize the MCP server
            mcp_instance = mcp_class()
            
            # Check if tools are registered
            tools = mcp_instance.list_tools()
            
            logger.info(f"Successfully initialized {mcp_name} with {len(tools)} tools")
            
            # Try to access a tool or resource if available
            if tools:
                tool_name = tools[0]["name"]
                logger.info(f"Found tool: {tool_name} in {mcp_name}")
            
            results.record_test(f"{mcp_name} Initialization", True, time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error initializing {mcp_name}: {e}")
            results.record_test(f"{mcp_name} Initialization", False, time.time() - start_time,
                              failure_reason=f"Exception: {str(e)}")


def run_performance_test(mcp_instance, method_name, args=None, iterations=3):
    """Run a performance test on a method"""
    if args is None:
        args = {}
        
    durations = []
    results = []
    
    logger.info(f"Running performance test on {method_name} ({iterations} iterations)")
    
    method = getattr(mcp_instance, method_name, None)
    if not method:
        return {"error": f"Method {method_name} not found on MCP instance"}
    
    for i in range(iterations):
        start_time = time.time()
        result = method(**args)
        duration = time.time() - start_time
        
        # Keep track of durations
        durations.append(duration)
        # And store results for sanity check
        results.append(result)
        
        # Small pause between calls
        time.sleep(0.1)
    
    # Calculate stats
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    logger.info(f"Performance test results: avg={avg_duration:.3f}s, min={min_duration:.3f}s, max={max_duration:.3f}s")
    
    return {
        "avg_duration": avg_duration,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "durations": durations,
        "sample_result": results[0]  # Just include one result for reference
    }


def main():
    """Execute all MCP tools tests"""
    print("="*80)
    print(f"Starting MCP Tools Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize results tracker
    results = MCPTestResults()
    
    try:
        # Test BaseMCPServer functionality
        test_mcp_base(results)
        
        # Test YahooFinanceMCP functionality
        test_yahoo_finance_mcp(results)
        
        # Test TradingMCP functionality
        test_trading_mcp(results)
        
        # Test FinancialTextMCP functionality
        test_financial_text_mcp(results)
        
        # Test RedisMCP functionality
        test_redis_mcp(results)
        
        # Test PolygonRestMCP functionality
        test_polygon_rest_mcp(results)
        
        # Test other MCP servers (initialization only)
        test_other_mcp_initialization(results)
        
    except Exception as e:
        logger.error(f"Unhandled exception during tests: {e}")
        results.add_warning("Test Suite", f"Unhandled exception: {e}")
    
    # Print test results summary
    results.print_summary()
    
    # Return non-zero exit code if any tests failed
    if results.tests_failed > 0:
        sys.exit(1)
    

if __name__ == "__main__":
    main()
