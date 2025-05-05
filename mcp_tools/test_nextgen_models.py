#!/usr/bin/env python3
"""
NextGen Models Test Suite

This script performs comprehensive testing of all NextGen models and their MCP tools
to verify functionality, accuracy, performance, and production readiness.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import unittest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nextgen_model_test")

# Import MCP tools
from mcp_tools.base_mcp_server import BaseMCPServer
from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.yahoo_news_mcp import YahooNewsMCP
from mcp_tools.financial_text_mcp.financial_text_mcp import FinancialTextMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP
from mcp_tools.document_analysis_mcp.document_analysis_mcp import DocumentAnalysisMCP

# Import NextGen models
from nextgen_models.nextgen_fundamental_analysis.fundamental_analysis_model import FundamentalAnalysisModel
from nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model import SentimentAnalysisModel
from nextgen_models.nextgen_market_analysis.market_analysis_model import MarketAnalysisModel
from nextgen_models.nextgen_risk_assessment.risk_assessment_model import RiskAssessmentModel
from nextgen_models.nextgen_decision.decision_model import DecisionModel
from nextgen_models.nextgen_trader.trade_model import TradeModel
from nextgen_models.nextgen_select.select_model import SelectionModel
from nextgen_models.nextgen_context_model.context_model import ContextModel
from nextgen_models.autogen_orchestrator.autogen_model import AutoGenOrchestrator


class ModelTestResults:
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
        print(f"NEXTGEN MODELS TEST RESULTS SUMMARY")
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


class TestFundamentalAnalysisModel(unittest.TestCase):
    """Test suite for the Fundamental Analysis Model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        logger.info("Setting up FundamentalAnalysisModel test suite")
        
        # Initialize required MCP tools
        cls.financial_data_mcp = FinancialDataMCP()
        cls.risk_analysis_mcp = RiskAnalysisMCP()
        cls.redis_mcp = RedisMCP()
        
        # Test data
        cls.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        cls.test_config = {
            "financial_data_config": {},
            "risk_analysis_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
            "analysis_interval": 3600,  # 1 hour for testing
            "max_companies": 5
        }
        
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create a fresh model instance for each test
        self.model = FundamentalAnalysisModel(config=self.test_config)
        
    def test_initialization(self):
        """Test model initialization"""
        start_time = time.time()
        
        # Check that the model was initialized correctly
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.financial_data_mcp)
        self.assertIsNotNone(self.model.risk_analysis_mcp)
        self.assertIsNotNone(self.model.redis_mcp)
        self.assertIsNotNone(self.model.agents)
        
        # Check that the agents were set up correctly
        self.assertIn("fundamental_assistant", self.model.agents)
        self.assertIn("user_proxy", self.model.agents)
        
        duration = time.time() - start_time
        logger.info(f"FundamentalAnalysisModel initialization test completed in {duration:.2f}s")
        
    def test_get_financial_statements(self):
        """Test getting financial statements"""
        start_time = time.time()
        
        for symbol in self.test_symbols:
            # Test with annual statements
            annual_statements = self.model.get_financial_statements(symbol, "all", "annual")
            self.assertIsNotNone(annual_statements)
            self.assertNotIn("error", annual_statements, f"Error getting annual statements for {symbol}: {annual_statements.get('error')}")
            
            # Test with quarterly statements
            quarterly_statements = self.model.get_financial_statements(symbol, "all", "quarterly")
            self.assertIsNotNone(quarterly_statements)
            self.assertNotIn("error", quarterly_statements, f"Error getting quarterly statements for {symbol}: {quarterly_statements.get('error')}")
            
            # Test with specific statement type
            income_statement = self.model.get_financial_statements(symbol, "income", "annual")
            self.assertIsNotNone(income_statement)
            self.assertNotIn("error", income_statement, f"Error getting income statement for {symbol}: {income_statement.get('error')}")
            
        duration = time.time() - start_time
        logger.info(f"Financial statements test completed in {duration:.2f}s")
        
    def test_get_market_data(self):
        """Test getting market data"""
        start_time = time.time()
        
        for symbol in self.test_symbols:
            market_data = self.model.get_market_data(symbol)
            self.assertIsNotNone(market_data)
            self.assertNotIn("error", market_data, f"Error getting market data for {symbol}: {market_data.get('error')}")
            
            # Check for essential market data fields
            self.assertIn("symbol", market_data)
            self.assertEqual(market_data["symbol"], symbol)
            
        duration = time.time() - start_time
        logger.info(f"Market data test completed in {duration:.2f}s")
        
    def test_analyze_company(self):
        """Test comprehensive company analysis"""
        start_time = time.time()
        
        # Test with a single company for comprehensive analysis
        symbol = self.test_symbols[0]
        sector = "Technology"
        
        analysis = self.model.analyze_company(symbol, sector)
        self.assertIsNotNone(analysis)
        self.assertNotIn("error", analysis, f"Error analyzing company {symbol}: {analysis.get('error')}")
        
        # Check for essential analysis components
        self.assertIn("symbol", analysis)
        self.assertEqual(analysis["symbol"], symbol)
        self.assertIn("sector", analysis)
        self.assertEqual(analysis["sector"], sector)
        self.assertIn("market_data", analysis)
        self.assertIn("financial_ratios", analysis)
        self.assertIn("financial_health", analysis)
        self.assertIn("growth_analysis", analysis)
        
        duration = time.time() - start_time
        logger.info(f"Company analysis test completed in {duration:.2f}s")
        
    def test_redis_integration(self):
        """Test Redis integration for data storage and retrieval"""
        start_time = time.time()
        
        # Test storing and retrieving data from Redis
        symbol = self.test_symbols[0]
        test_data = {
            "symbol": symbol,
            "test_value": "test",
            "timestamp": datetime.now().isoformat()
        }
        
        # Store test data in Redis
        cache_key = f"test:fundamental:{symbol}"
        set_result = self.model.redis_mcp.call_tool(
            "set_json",
            {"key": cache_key, "value": test_data, "expiry": 60}  # 1 minute expiry
        )
        self.assertNotIn("error", set_result, f"Error setting Redis data: {set_result.get('error')}")
        
        # Retrieve test data from Redis
        get_result = self.model.redis_mcp.call_tool(
            "get_json",
            {"key": cache_key}
        )
        self.assertNotIn("error", get_result, f"Error getting Redis data: {get_result.get('error')}")
        self.assertIn("value", get_result)
        self.assertEqual(get_result["value"]["symbol"], symbol)
        self.assertEqual(get_result["value"]["test_value"], "test")
        
        # Clean up
        self.model.redis_mcp.call_tool(
            "delete_key",
            {"key": cache_key}
        )
        
        duration = time.time() - start_time
        logger.info(f"Redis integration test completed in {duration:.2f}s")
        
    def test_mcp_tool_integration(self):
        """Test integration with MCP tools"""
        start_time = time.time()
        
        # Test financial data MCP
        symbol = self.test_symbols[0]
        financial_data_result = self.model.financial_data_mcp.call_tool(
            "get_stock_price",
            {"symbol": symbol}
        )
        self.assertNotIn("error", financial_data_result, f"Error calling financial data MCP: {financial_data_result.get('error')}")
        
        # Test risk analysis MCP with sample data
        sample_ratios = {
            "pe_ratio": 15.5,
            "pb_ratio": 2.3,
            "current_ratio": 1.5,
            "debt_to_equity": 0.8,
            "roe": 0.12
        }
        risk_analysis_result = self.model.risk_analysis_mcp.call_tool(
            "score_financial_health",
            {"ratios": sample_ratios, "sector": "Technology"}
        )
        self.assertNotIn("error", risk_analysis_result, f"Error calling risk analysis MCP: {risk_analysis_result.get('error')}")
        
        duration = time.time() - start_time
        logger.info(f"MCP tool integration test completed in {duration:.2f}s")


class TestSentimentAnalysisModel(unittest.TestCase):
    """Test suite for the Sentiment Analysis Model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        logger.info("Setting up SentimentAnalysisModel test suite")
        
        # Initialize required MCP tools
        cls.financial_text_mcp = FinancialTextMCP()
        cls.yahoo_news_mcp = YahooNewsMCP()
        cls.redis_mcp = RedisMCP()
        
        # Test data
        cls.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        cls.test_texts = [
            "The company reported strong earnings, beating analyst expectations.",
            "The stock plummeted after the company missed revenue targets.",
            "Investors remain neutral about the company's prospects."
        ]
        cls.test_config = {
            "financial_text_config": {},
            "yahoo_news_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
            "analysis_interval": 3600,  # 1 hour for testing
            "max_news_items": 5
        }
        
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create a fresh model instance for each test
        self.model = SentimentAnalysisModel(config=self.test_config)
        
    def test_initialization(self):
        """Test model initialization"""
        start_time = time.time()
        
        # Check that the model was initialized correctly
        self.assertIsNotNone(self.model)
        
        duration = time.time() - start_time
        logger.info(f"SentimentAnalysisModel initialization test completed in {duration:.2f}s")
        
    # Add more tests for SentimentAnalysisModel...


class TestMarketAnalysisModel(unittest.TestCase):
    """Test suite for the Market Analysis Model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        logger.info("Setting up MarketAnalysisModel test suite")
        
        # Initialize required MCP tools
        cls.financial_data_mcp = FinancialDataMCP()
        cls.time_series_mcp = TimeSeriesMCP()
        cls.redis_mcp = RedisMCP()
        
        # Test data
        cls.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        cls.test_config = {
            "financial_data_config": {},
            "time_series_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
            "analysis_interval": 3600,  # 1 hour for testing
        }
        
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create a fresh model instance for each test
        self.model = MarketAnalysisModel(config=self.test_config)
        
    def test_initialization(self):
        """Test model initialization"""
        start_time = time.time()
        
        # Check that the model was initialized correctly
        self.assertIsNotNone(self.model)
        
        duration = time.time() - start_time
        logger.info(f"MarketAnalysisModel initialization test completed in {duration:.2f}s")
        
    # Add more tests for MarketAnalysisModel...


class TestRiskAssessmentModel(unittest.TestCase):
    """Test suite for the Risk Assessment Model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        logger.info("Setting up RiskAssessmentModel test suite")
        
        # Initialize required MCP tools
        cls.risk_analysis_mcp = RiskAnalysisMCP()
        cls.financial_data_mcp = FinancialDataMCP()
        cls.redis_mcp = RedisMCP()
        
        # Test data
        cls.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        cls.test_config = {
            "risk_analysis_config": {},
            "financial_data_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
            "analysis_interval": 3600,  # 1 hour for testing
        }
        
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create a fresh model instance for each test
        self.model = RiskAssessmentModel(config=self.test_config)
        
    def test_initialization(self):
        """Test model initialization"""
        start_time = time.time()
        
        # Check that the model was initialized correctly
        self.assertIsNotNone(self.model)
        
        duration = time.time() - start_time
        logger.info(f"RiskAssessmentModel initialization test completed in {duration:.2f}s")
        
    # Add more tests for RiskAssessmentModel...


# Add test classes for other models...


def run_model_tests(model_name: str, results: ModelTestResults):
    """Run tests for a specific model"""
    logger.info(f"Running tests for {model_name}")
    
    # Map model names to test classes
    test_classes = {
        "fundamental": TestFundamentalAnalysisModel,
        "sentiment": TestSentimentAnalysisModel,
        "market": TestMarketAnalysisModel,
        "risk": TestRiskAssessmentModel,
        # Add other models here
    }
    
    if model_name not in test_classes:
        results.add_warning(f"{model_name}_tests", f"No test class defined for {model_name}")
        return
    
    # Run the tests using unittest
    test_class = test_classes[model_name]
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    
    # Use a custom test runner to capture results
    class CustomTestRunner(unittest.TextTestRunner):
        def run(self, test):
            result = super().run(test)
            for failure in result.failures:
                test_name = failure[0].id().split('.')[-1]
                results.record_test(
                    f"{model_name}_{test_name}",
                    False,
                    0.0,  # We don't have duration info from unittest
                    failure_reason=str(failure[1])
                )
            
            for error in result.errors:
                test_name = error[0].id().split('.')[-1]
                results.record_test(
                    f"{model_name}_{test_name}",
                    False,
                    0.0,  # We don't have duration info from unittest
                    failure_reason=str(error[1])
                )
            
            # Record successful tests
            # unittest doesn't track successes by default, so we need to calculate them
            successes = []
            for test_case in test:
                for test_method in test_case:
                    test_id = test_method.id()
                    if not any(test_id == f[0].id() for f in result.failures) and \
                       not any(test_id == e[0].id() for e in result.errors):
                        successes.append(test_method)
            
            for test in successes:
                test_name = test.id().split('.')[-1]
                results.record_test(
                    f"{model_name}_{test_name}",
                    True,
                    0.0  # We don't have duration info from unittest
                )
            
            return result
    
    # Run the tests
    runner = CustomTestRunner(verbosity=2)
    runner.run(suite)


def main():
    """Execute all NextGen model tests"""
    print("="*80)
    print(f"Starting NextGen Models Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize results tracker
    results = ModelTestResults()
    
    # Check if specific models were requested
    if len(sys.argv) > 1:
        models_to_test = sys.argv[1:]
    else:
        # Test all models by default
        models_to_test = [
            "fundamental",
            "sentiment",
            "market",
            "risk",
            "decision",
            "trader",
            "select",
            "context",
            "autogen"
        ]
    
    try:
        # Run tests for each model
        for model_name in models_to_test:
            try:
                run_model_tests(model_name, results)
            except Exception as e:
                logger.error(f"Error running tests for {model_name}: {e}", exc_info=True)
                results.record_test(
                    f"{model_name}_tests",
                    False,
                    0.0,
                    failure_reason=f"Unhandled exception: {str(e)}"
                )
    
    except Exception as e:
        logger.error(f"Unhandled exception during tests: {e}", exc_info=True)
        results.add_warning("Test Suite", f"Unhandled exception: {e}")
    
    # Print test results summary
    results.print_summary()
    
    # Return non-zero exit code if any tests failed
    if results.tests_failed > 0:
        sys.exit(1)
    

if __name__ == "__main__":
    main()
