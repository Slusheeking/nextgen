#!/usr/bin/env python3
"""
NextGen Model Accuracy Test Suite

This script performs comprehensive testing of all NextGen models and their MCP tools
with appropriate datasets to ensure accuracy, proper LLM integration, and correct
functionality of associated MCP tools.

Usage:
    python test_model_accuracy.py [model_name]
    
    If no model name is provided, all models will be tested sequentially.
    Available models: fundamental, sentiment, market, risk, decision, trader, select, context, autogen
"""

import os
import sys
import time
import json
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Add the project root directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_accuracy_test")

# Import MCP tools with error handling
mcp_imports_available = {}

# Function to safely import modules
def safe_import(module_name, class_name=None):
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import {module_name}{f'.{class_name}' if class_name else ''}: {e}")
        return None

# Base MCP Server
BaseMCPServer = safe_import("mcp_tools.base_mcp_server", "BaseMCPServer")
mcp_imports_available["base_mcp"] = BaseMCPServer is not None

# Data MCP tools
YahooFinanceMCP = safe_import("mcp_tools.data_mcp.yahoo_finance_mcp", "YahooFinanceMCP")
mcp_imports_available["yahoo_finance"] = YahooFinanceMCP is not None

PolygonRestMCP = safe_import("mcp_tools.data_mcp.polygon_rest_mcp", "PolygonRestMCP")
mcp_imports_available["polygon_rest"] = PolygonRestMCP is not None

YahooNewsMCP = safe_import("mcp_tools.data_mcp.yahoo_news_mcp", "YahooNewsMCP")
mcp_imports_available["yahoo_news"] = YahooNewsMCP is not None

# Financial text MCP
FinancialTextMCP = safe_import("mcp_tools.financial_text_mcp.financial_text_mcp", "FinancialTextMCP")
mcp_imports_available["financial_text"] = FinancialTextMCP is not None

# Database MCP
RedisMCP = safe_import("mcp_tools.db_mcp.redis_mcp", "RedisMCP")
mcp_imports_available["redis"] = RedisMCP is not None

# Financial data MCP
FinancialDataMCP = safe_import("mcp_tools.financial_data_mcp.financial_data_mcp", "FinancialDataMCP")
mcp_imports_available["financial_data"] = FinancialDataMCP is not None

# Risk analysis MCP
RiskAnalysisMCP = safe_import("mcp_tools.risk_analysis_mcp.risk_analysis_mcp", "RiskAnalysisMCP")
mcp_imports_available["risk_analysis"] = RiskAnalysisMCP is not None

# Time series MCP
try:
    from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
    mcp_imports_available["time_series"] = True
except (ImportError, AttributeError) as e:
    logger.warning(f"Could not import TimeSeriesMCP: {e}")
    TimeSeriesMCP = None
    mcp_imports_available["time_series"] = False

# Vector store MCP
VectorStoreMCP = safe_import("mcp_tools.vector_store_mcp.vector_store_mcp", "VectorStoreMCP")
mcp_imports_available["vector_store"] = VectorStoreMCP is not None

# Document analysis MCP
DocumentAnalysisMCP = safe_import("mcp_tools.document_analysis_mcp.document_analysis_mcp", "DocumentAnalysisMCP")
mcp_imports_available["document_analysis"] = DocumentAnalysisMCP is not None

# Import NextGen models with error handling
models_available = {}

# Fundamental Analysis Model
try:
    from nextgen_models.nextgen_fundamental_analysis.fundamental_analysis_model import FundamentalAnalysisModel
    models_available["fundamental"] = True
except ImportError as e:
    logger.warning(f"Could not import FundamentalAnalysisModel: {e}")
    FundamentalAnalysisModel = None
    models_available["fundamental"] = False

# Sentiment Analysis Model
try:
    from nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model import SentimentAnalysisModel
    models_available["sentiment"] = True
except ImportError as e:
    logger.warning(f"Could not import SentimentAnalysisModel: {e}")
    SentimentAnalysisModel = None
    models_available["sentiment"] = False

# Market Analysis Model
try:
    from nextgen_models.nextgen_market_analysis.market_analysis_model import MarketAnalysisModel
    models_available["market"] = True
except ImportError as e:
    logger.warning(f"Could not import MarketAnalysisModel: {e}")
    MarketAnalysisModel = None
    models_available["market"] = False

# Risk Assessment Model
try:
    from nextgen_models.nextgen_risk_assessment.risk_assessment_model import RiskAssessmentModel
    models_available["risk"] = True
except ImportError as e:
    logger.warning(f"Could not import RiskAssessmentModel: {e}")
    RiskAssessmentModel = None
    models_available["risk"] = False

# Other models
try:
    from nextgen_models.nextgen_decision.decision_model import DecisionModel
    models_available["decision"] = True
except ImportError as e:
    logger.warning(f"Could not import DecisionModel: {e}")
    DecisionModel = None
    models_available["decision"] = False

try:
    from nextgen_models.nextgen_trader.trade_model import TradeModel
    models_available["trader"] = True
except ImportError as e:
    logger.warning(f"Could not import TradeModel: {e}")
    TradeModel = None
    models_available["trader"] = False

try:
    from nextgen_models.nextgen_select.select_model import SelectionModel
    models_available["select"] = True
except ImportError as e:
    logger.warning(f"Could not import SelectionModel: {e}")
    SelectionModel = None
    models_available["select"] = False

try:
    from nextgen_models.nextgen_context_model.context_model import ContextModel
    models_available["context"] = True
except ImportError as e:
    logger.warning(f"Could not import ContextModel: {e}")
    ContextModel = None
    models_available["context"] = False

try:
    from nextgen_models.autogen_orchestrator.autogen_model import AutoGenOrchestrator
    models_available["autogen"] = True
except ImportError as e:
    logger.warning(f"Could not import AutoGenOrchestrator: {e}")
    AutoGenOrchestrator = None
    models_available["autogen"] = False

# Test datasets
TEST_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
TEST_SECTORS = ["Technology", "Consumer Cyclical", "Communication Services"]
TEST_NEWS_ITEMS = [
    "Company reports record profits, beating analyst expectations by 15%.",
    "CEO announces resignation amid accounting scandal investigation.",
    "New product launch receives mixed reviews from industry experts.",
    "Company announces major layoffs affecting 10% of workforce.",
    "Quarterly earnings show steady growth in line with market expectations."
]
TEST_MARKET_CONDITIONS = ["bull", "bear", "neutral", "volatile"]


class AccuracyTestResults:
    """Class to track and report test results with accuracy metrics"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.warnings = []
        self.performance_metrics = {}
        self.accuracy_metrics = {}
        
    def record_test(self, test_name: str, success: bool, duration: float, 
                   accuracy: Optional[float] = None,
                   details: Optional[Dict[str, Any]] = None, 
                   failure_reason: Optional[str] = None):
        """Record result of a single test with accuracy metrics"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            logger.info(f"✅ PASS: {test_name} ({duration:.2f}s)" + 
                       (f", Accuracy: {accuracy:.2f}%" if accuracy is not None else ""))
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
        
        # Record accuracy if provided
        if accuracy is not None:
            if test_name not in self.accuracy_metrics:
                self.accuracy_metrics[test_name] = []
            self.accuracy_metrics[test_name].append(accuracy)
    
    def add_warning(self, test_name: str, warning: str):
        """Add a warning message"""
        logger.warning(f"⚠️ WARNING: {test_name} - {warning}")
        self.warnings.append({
            "test": test_name,
            "warning": warning
        })
        
    def print_summary(self):
        """Print test results summary with accuracy metrics"""
        print("\n" + "="*80)
        print(f"NEXTGEN MODELS ACCURACY TEST RESULTS SUMMARY")
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
        
        print("\nACCURACY SUMMARY:")
        for test, accuracies in self.accuracy_metrics.items():
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            print(f"- {test}: Avg Accuracy {avg_accuracy:.2f}% (min: {min(accuracies):.2f}%, max: {max(accuracies):.2f}%)")
            
        print("="*80)
        
    def get_overall_status(self):
        """Get overall test status"""
        if self.tests_failed > 0:
            return "FAILED"
        elif self.warnings:
            return "PASSED WITH WARNINGS"
        else:
            return "PASSED"


class ModelAccuracyTester:
    """Base class for testing model accuracy"""
    
    def __init__(self, model_name: str, results: AccuracyTestResults):
        self.model_name = model_name
        self.results = results
        self.model = None
        self.test_data = {}
        self.expected_results = {}
        
    def setup(self):
        """Set up the test environment"""
        logger.info(f"Setting up test environment for {self.model_name}")
        
    def teardown(self):
        """Clean up after tests"""
        logger.info(f"Cleaning up after {self.model_name} tests")
        
    def prepare_test_data(self):
        """Prepare test data for the model"""
        logger.info(f"Preparing test data for {self.model_name}")
        
    def run_tests(self):
        """Run all tests for this model"""
        logger.info(f"Running tests for {self.model_name}")
        
    def verify_mcp_tools(self):
        """Verify that all required MCP tools are working"""
        logger.info(f"Verifying MCP tools for {self.model_name}")
        
    def test_llm_integration(self):
        """Test LLM integration"""
        logger.info(f"Testing LLM integration for {self.model_name}")
        
    def calculate_accuracy(self, actual: Any, expected: Any) -> float:
        """
        Calculate accuracy between actual and expected results
        
        This is a base implementation that should be overridden by subclasses
        to provide model-specific accuracy calculations.
        """
        if isinstance(actual, dict) and isinstance(expected, dict):
            # For dictionaries, calculate percentage of matching keys
            matching_keys = sum(1 for k in expected if k in actual and actual[k] == expected[k])
            return (matching_keys / len(expected)) * 100 if expected else 0
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            # For lists, calculate percentage of matching elements
            matching_elements = sum(1 for a, e in zip(actual, expected) if a == e)
            return (matching_elements / len(expected)) * 100 if expected else 0
        else:
            # For other types, return 100 if equal, 0 otherwise
            return 100 if actual == expected else 0


class FundamentalAnalysisModelTester(ModelAccuracyTester):
    """Test suite for the Fundamental Analysis Model"""
    
    def __init__(self, results: AccuracyTestResults):
        super().__init__("Fundamental Analysis Model", results)
        self.financial_data_mcp = None
        self.risk_analysis_mcp = None
        self.redis_mcp = None
        
    def setup(self):
        """Set up the test environment"""
        super().setup()
        
        # Initialize required MCP tools
        self.financial_data_mcp = FinancialDataMCP()
        self.risk_analysis_mcp = RiskAnalysisMCP()
        self.redis_mcp = RedisMCP()
        
        # Test config with API keys from environment
        self.test_config = {
            "financial_data_config": {},
            "risk_analysis_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": os.environ.get("LLM_MODEL", "anthropic/claude-3-opus-20240229"),
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
        
        # Initialize the model
        try:
            self.model = FundamentalAnalysisModel(config=self.test_config)
            logger.info("Fundamental Analysis Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Fundamental Analysis Model: {e}")
            self.results.record_test(
                "fundamental_model_init",
                False,
                0.0,
                failure_reason=f"Initialization failed: {str(e)}"
            )
            
    def prepare_test_data(self):
        """Prepare test data for the model"""
        super().prepare_test_data()
        
        # Prepare test stocks
        self.test_data["symbols"] = TEST_STOCKS[:3]  # Use first 3 test stocks
        self.test_data["sectors"] = TEST_SECTORS[:3]  # Use first 3 test sectors
        
        # Create expected results structure (will be populated during tests)
        self.expected_results = {
            "financial_statements": {},
            "market_data": {},
            "company_analysis": {}
        }
        
    def verify_mcp_tools(self):
        """Verify that all required MCP tools are working"""
        super().verify_mcp_tools()
        
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            # Test financial data MCP
            symbol = self.test_data["symbols"][0]
            financial_data_result = self.financial_data_mcp.call_tool(
                "get_stock_price",
                {"symbol": symbol}
            )
            
            if "error" in financial_data_result:
                success = False
                failure_reason = f"Financial Data MCP error: {financial_data_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info(f"Financial Data MCP successfully retrieved stock price for {symbol}")
            
            # Test risk analysis MCP
            sample_returns = {
                "AAPL": [0.01, -0.005, 0.02, -0.01, 0.015],
                "MSFT": [0.008, -0.003, 0.015, -0.007, 0.01]
            }
            
            risk_analysis_result = self.risk_analysis_mcp.call_tool(
                "calculate_risk_metrics",
                {"returns_data": sample_returns}
            )
            
            if "error" in risk_analysis_result:
                success = False
                failure_reason = f"Risk Analysis MCP error: {risk_analysis_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info("Risk Analysis MCP successfully calculated risk metrics")
            
            # Test Redis MCP
            test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
            redis_set_result = self.redis_mcp.call_tool(
                "set_json",
                {"key": "test:fundamental", "value": test_data, "expiry": 60}
            )
            
            if "error" in redis_set_result:
                success = False
                failure_reason = f"Redis MCP set error: {redis_set_result['error']}"
                logger.error(failure_reason)
            else:
                # Try to get the data back
                redis_get_result = self.redis_mcp.call_tool(
                    "get_json",
                    {"key": "test:fundamental"}
                )
                
                if "error" in redis_get_result:
                    success = False
                    failure_reason = f"Redis MCP get error: {redis_get_result['error']}"
                    logger.error(failure_reason)
                else:
                    logger.info("Redis MCP successfully stored and retrieved data")
                    
                # Clean up
                self.redis_mcp.call_tool(
                    "delete_key",
                    {"key": "test:fundamental"}
                )
                
        except Exception as e:
            success = False
            failure_reason = f"MCP tool verification failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "fundamental_mcp_tools",
            success,
            duration,
            failure_reason=failure_reason
        )
        
    def test_financial_statements(self):
        """Test retrieving financial statements"""
        if not self.model:
            self.results.add_warning("fundamental_financial_statements", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            for symbol in self.test_data["symbols"]:
                # Test with annual statements
                annual_statements = self.model.get_financial_statements(symbol, "all", "annual")
                
                if "error" in annual_statements:
                    success = False
                    failure_reason = f"Error getting annual statements for {symbol}: {annual_statements['error']}"
                    logger.error(failure_reason)
                    break
                
                # Test with quarterly statements
                quarterly_statements = self.model.get_financial_statements(symbol, "all", "quarterly")
                
                if "error" in quarterly_statements:
                    success = False
                    failure_reason = f"Error getting quarterly statements for {symbol}: {quarterly_statements['error']}"
                    logger.error(failure_reason)
                    break
                
                # Store results for accuracy calculation
                self.expected_results["financial_statements"][symbol] = {
                    "annual": annual_statements,
                    "quarterly": quarterly_statements
                }
                
                logger.info(f"Successfully retrieved financial statements for {symbol}")
                
        except Exception as e:
            success = False
            failure_reason = f"Financial statements test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "fundamental_financial_statements",
            success,
            duration,
            failure_reason=failure_reason
        )
        
    def test_market_data(self):
        """Test retrieving market data"""
        if not self.model:
            self.results.add_warning("fundamental_market_data", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            for symbol in self.test_data["symbols"]:
                market_data = self.model.get_market_data(symbol)
                
                if "error" in market_data:
                    success = False
                    failure_reason = f"Error getting market data for {symbol}: {market_data['error']}"
                    logger.error(failure_reason)
                    break
                
                # Check for essential market data fields
                required_fields = ["symbol", "price", "volume", "market_cap"]
                missing_fields = [field for field in required_fields if field not in market_data]
                
                if missing_fields:
                    success = False
                    failure_reason = f"Missing required fields in market data for {symbol}: {missing_fields}"
                    logger.error(failure_reason)
                    break
                
                # Store results for accuracy calculation
                self.expected_results["market_data"][symbol] = market_data
                
                logger.info(f"Successfully retrieved market data for {symbol}")
                
        except Exception as e:
            success = False
            failure_reason = f"Market data test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "fundamental_market_data",
            success,
            duration,
            failure_reason=failure_reason
        )
        
    def test_company_analysis(self):
        """Test comprehensive company analysis"""
        if not self.model:
            self.results.add_warning("fundamental_company_analysis", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        accuracy = None
        
        try:
            # Test with a single company for comprehensive analysis
            symbol = self.test_data["symbols"][0]
            sector = self.test_data["sectors"][0]
            
            analysis = self.model.analyze_company(symbol, sector)
            
            if "error" in analysis:
                success = False
                failure_reason = f"Error analyzing company {symbol}: {analysis['error']}"
                logger.error(failure_reason)
            else:
                # Check for essential analysis components
                required_components = [
                    "symbol", "sector", "market_data", "financial_ratios", 
                    "financial_health", "growth_analysis"
                ]
                
                missing_components = [comp for comp in required_components if comp not in analysis]
                
                if missing_components:
                    success = False
                    failure_reason = f"Missing required components in analysis for {symbol}: {missing_components}"
                    logger.error(failure_reason)
                else:
                    # Store results for accuracy calculation
                    self.expected_results["company_analysis"][symbol] = analysis
                    
                    # Calculate accuracy based on completeness of analysis
                    total_components = len(required_components)
                    present_components = sum(1 for comp in required_components if comp in analysis)
                    accuracy = (present_components / total_components) * 100
                    
                    logger.info(f"Successfully analyzed company {symbol} with {accuracy:.2f}% completeness")
                
        except Exception as e:
            success = False
            failure_reason = f"Company analysis test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "fundamental_company_analysis",
            success,
            duration,
            accuracy=accuracy,
            failure_reason=failure_reason
        )
        
    def test_llm_integration(self):
        """Test LLM integration for fundamental analysis"""
        if not self.model:
            self.results.add_warning("fundamental_llm_integration", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        accuracy = None
        
        try:
            # Test LLM-based analysis with a simple query
            symbol = self.test_data["symbols"][0]
            
            # First, ensure we have market data and financial statements
            if symbol not in self.expected_results["market_data"] or symbol not in self.expected_results["financial_statements"]:
                self.results.add_warning("fundamental_llm_integration", "Missing prerequisite data, retrieving now")
                
                if symbol not in self.expected_results["market_data"]:
                    self.expected_results["market_data"][symbol] = self.model.get_market_data(symbol)
                    
                if symbol not in self.expected_results["financial_statements"]:
                    self.expected_results["financial_statements"][symbol] = {
                        "annual": self.model.get_financial_statements(symbol, "all", "annual"),
                        "quarterly": self.model.get_financial_statements(symbol, "all", "quarterly")
                    }
            
            # Prepare input data for LLM
            market_data = self.expected_results["market_data"][symbol]
            financial_data = self.expected_results["financial_statements"][symbol]["annual"]
            
            # Create a simple analysis request
            analysis_request = {
                "symbol": symbol,
                "market_data": market_data,
                "financial_data": financial_data,
                "query": f"Provide a brief financial health assessment of {symbol} based on the provided data."
            }
            
            # Use the model's LLM to analyze the data
            llm_analysis = self.model.analyze_with_llm(analysis_request)
            
            if not llm_analysis or "error" in llm_analysis:
                success = False
                failure_reason = f"Error in LLM analysis: {llm_analysis.get('error', 'No result returned')}"
                logger.error(failure_reason)
            else:
                # Check for minimum content length as a basic quality check
                min_length = 100  # Expect at least 100 characters
                if len(llm_analysis) < min_length:
                    success = False
                    failure_reason = f"LLM analysis too short: {len(llm_analysis)} chars (expected {min_length})"
                    logger.error(failure_reason)
                else:
                    # Calculate a simple accuracy metric based on content relevance
                    # Check if the analysis mentions the company symbol and key financial terms
                    key_terms = [symbol, "revenue", "profit", "growth", "ratio", "financial"]
                    term_matches = sum(1 for term in key_terms if term.lower() in llm_analysis.lower())
                    accuracy = (term_matches / len(key_terms)) * 100
                    
                    logger.info(f"LLM analysis completed with {accuracy:.2f}% relevance score")
                    logger.info(f"Analysis excerpt: {llm_analysis[:100]}...")
                
        except Exception as e:
            success = False
            failure_reason = f"LLM integration test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "fundamental_llm_integration",
            success,
            duration,
            accuracy=accuracy,
            failure_reason=failure_reason
        )
        
    def run_tests(self):
        """Run all tests for this model"""
        super().run_tests()
        
        # Skip tests if model initialization failed
        if not self.model:
            self.results.add_warning("fundamental_tests", "Model not initialized, skipping tests")
            return
            
        # Prepare test data
        self.prepare_test_data()
        
        # Run tests
        self.verify_mcp_tools()
        self.test_financial_statements()
        self.test_market_data()
        self.test_company_analysis()
        self.test_llm_integration()
        
    def teardown(self):
        """Clean up after tests"""
        super().teardown()
        
        # Clean up Redis test data if needed
        if self.redis_mcp:
            for symbol in self.test_data.get("symbols", []):
                self.redis_mcp.call_tool(
                    "delete_key",
                    {"key": f"test:fundamental:{symbol}"}
                )


class SentimentAnalysisModelTester(ModelAccuracyTester):
    """Test suite for the Sentiment Analysis Model"""
    
    def __init__(self, results: AccuracyTestResults):
        super().__init__("Sentiment Analysis Model", results)
        self.financial_text_mcp = None
        self.yahoo_news_mcp = None
        self.redis_mcp = None
        
    def setup(self):
        """Set up the test environment"""
        super().setup()
        
        # Initialize required MCP tools
        self.financial_text_mcp = FinancialTextMCP()
        self.yahoo_news_mcp = YahooNewsMCP()
        self.redis_mcp = RedisMCP()
        
        # Test config with API keys from environment
        self.test_config = {
            "financial_text_config": {},
            "yahoo_news_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": os.environ.get("LLM_MODEL", "anthropic/claude-3-opus-20240229"),
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
        
        # Initialize the model
        try:
            self.model = SentimentAnalysisModel(config=self.test_config)
            logger.info("Sentiment Analysis Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analysis Model: {e}")
            self.results.record_test(
                "sentiment_model_init",
                False,
                0.0,
                failure_reason=f"Initialization failed: {str(e)}"
            )
            
    def prepare_test_data(self):
        """Prepare test data for the model"""
        super().prepare_test_data()
        
        # Prepare test stocks and news items
        self.test_data["symbols"] = TEST_STOCKS[:3]  # Use first 3 test stocks
        self.test_data["news_items"] = TEST_NEWS_ITEMS
        
        # Create expected results structure (will be populated during tests)
        self.expected_results = {
            "news_sentiment": {},
            "text_sentiment": {},
            "company_sentiment": {}
        }
        
        # Create ground truth sentiment labels for test news items
        # Format: (positive_score, negative_score, neutral_score)
        self.test_data["ground_truth"] = {
            TEST_NEWS_ITEMS[0]: (0.8, 0.1, 0.1),  # Positive
            TEST_NEWS_ITEMS[1]: (0.1, 0.8, 0.1),  # Negative
            TEST_NEWS_ITEMS[2]: (0.4, 0.3, 0.3),  # Mixed
            TEST_NEWS_ITEMS[3]: (0.1, 0.7, 0.2),  # Negative
            TEST_NEWS_ITEMS[4]: (0.5, 0.2, 0.3)   # Slightly positive
        }
        
    def verify_mcp_tools(self):
        """Verify that all required MCP tools are working"""
        super().verify_mcp_tools()
        
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            # Test financial text MCP
            test_text = self.test_data["news_items"][0]
            text_analysis_result = self.financial_text_mcp.call_tool(
                "analyze_sentiment",
                {"text": test_text}
            )
            
            if "error" in text_analysis_result:
                success = False
                failure_reason = f"Financial Text MCP error: {text_analysis_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info("Financial Text MCP successfully analyzed sentiment")
            
            # Test Yahoo News MCP
            symbol = self.test_data["symbols"][0]
            news_result = self.yahoo_news_mcp.call_tool(
                "get_company_news",
                {"symbol": symbol, "limit": 3}
            )
            
            if "error" in news_result:
                success = False
                failure_reason = f"Yahoo News MCP error: {news_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info(f"Yahoo News MCP successfully retrieved news for {symbol}")
            
            # Test Redis MCP
            test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
            redis_set_result = self.redis_mcp.call_tool(
                "set_json",
                {"key": "test:sentiment", "value": test_data, "expiry": 60}
            )
            
            if "error" in redis_set_result:
                success = False
                failure_reason = f"Redis MCP set error: {redis_set_result['error']}"
                logger.error(failure_reason)
            else:
                # Try to get the data back
                redis_get_result = self.redis_mcp.call_tool(
                    "get_json",
                    {"key": "test:sentiment"}
                )
                
                if "error" in redis_get_result:
                    success = False
                    failure_reason = f"Redis MCP get error: {redis_get_result['error']}"
                    logger.error(failure_reason)
                else:
                    logger.info("Redis MCP successfully stored and retrieved data")
                    
                # Clean up
                self.redis_mcp.call_tool(
                    "delete_key",
                    {"key": "test:sentiment"}
                )
                
        except Exception as e:
            success = False
            failure_reason = f"MCP tool verification failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "sentiment_mcp_tools",
            success,
            duration,
            failure_reason=failure_reason
        )
        
    def test_text_sentiment(self):
        """Test sentiment analysis on individual text items"""
        if not self.model:
            self.results.add_warning("sentiment_text_analysis", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        accuracy = None
        
        try:
            # Test sentiment analysis on each test news item
            sentiment_results = {}
            accuracy_scores = []
            
            for text in self.test_data["news_items"]:
                # Analyze sentiment
                sentiment = self.model.analyze_text_sentiment(text)
                
                if "error" in sentiment:
                    success = False
                    failure_reason = f"Error analyzing sentiment: {sentiment['error']}"
                    logger.error(failure_reason)
                    break
                
                # Store result
                sentiment_results[text] = sentiment
                
                # Calculate accuracy against ground truth if available
                if text in self.test_data["ground_truth"]:
                    ground_truth = self.test_data["ground_truth"][text]
                    
                    # Extract sentiment scores from result
                    pos_score = sentiment.get("positive", 0)
                    neg_score = sentiment.get("negative", 0)
                    neu_score = sentiment.get("neutral", 0)
                    
                    # Calculate accuracy as cosine similarity between predicted and ground truth
                    pred_vector = [pos_score, neg_score, neu_score]
                    true_vector = list(ground_truth)
                    
                    # Normalize vectors
                    pred_norm = np.sqrt(sum(x*x for x in pred_vector))
                    true_norm = np.sqrt(sum(x*x for x in true_vector))
                    
                    if pred_norm > 0 and true_norm > 0:
                        # Calculate cosine similarity
                        similarity = sum(p*t for p, t in zip(pred_vector, true_vector)) / (pred_norm * true_norm)
                        # Convert to percentage
                        accuracy_score = similarity * 100
                        accuracy_scores.append(accuracy_score)
                        
                        logger.info(f"Sentiment accuracy for text: {accuracy_score:.2f}%")
            
            # Store results for later use
            self.expected_results["text_sentiment"] = sentiment_results
            
            # Calculate overall accuracy
            if accuracy_scores:
                accuracy = sum(accuracy_scores) / len(accuracy_scores)
                logger.info(f"Overall sentiment analysis accuracy: {accuracy:.2f}%")
                
        except Exception as e:
            success = False
            failure_reason = f"Text sentiment analysis test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "sentiment_text_analysis",
            success,
            duration,
            accuracy=accuracy,
            failure_reason=failure_reason
        )
        
    def test_company_sentiment(self):
        """Test sentiment analysis for companies"""
        if not self.model:
            self.results.add_warning("sentiment_company_analysis", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            # Test company sentiment analysis for each test symbol
            for symbol in self.test_data["symbols"]:
                # Analyze company sentiment
                sentiment = self.model.analyze_company_sentiment(symbol)
                
                if "error" in sentiment:
                    success = False
                    failure_reason = f"Error analyzing company sentiment for {symbol}: {sentiment['error']}"
                    logger.error(failure_reason)
                    break
                
                # Check for required fields
                required_fields = ["symbol", "overall_sentiment", "news_sentiment", "social_sentiment"]
                missing_fields = [field for field in required_fields if field not in sentiment]
                
                if missing_fields:
                    success = False
                    failure_reason = f"Missing required fields in company sentiment for {symbol}: {missing_fields}"
                    logger.error(failure_reason)
                    break
                
                # Store result
                self.expected_results["company_sentiment"][symbol] = sentiment
                
                logger.info(f"Successfully analyzed sentiment for {symbol}")
                
        except Exception as e:
            success = False
            failure_reason = f"Company sentiment analysis test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "sentiment_company_analysis",
            success,
            duration,
            failure_reason=failure_reason
        )
        
    def test_llm_integration(self):
        """Test LLM integration for sentiment analysis"""
        if not self.model:
            self.results.add_warning("sentiment_llm_integration", "Model not initialized, skipping test")
            return
            
        start_time = time.time()
        success = True
        failure_reason = None
        accuracy = None
        
        try:
            # Test LLM-based sentiment analysis with a complex text
            complex_text = """
            The company's Q2 earnings showed mixed results. While revenue increased by 5% year-over-year,
            profit margins decreased slightly due to increased competition. The CEO expressed confidence
            in the company's long-term strategy despite short-term challenges. Analysts have mixed opinions,
            with some maintaining buy ratings while others have downgraded to hold.
            """
            
            # Use the model's LLM to analyze sentiment
            llm_sentiment = self.model.analyze_with_llm({
                "text": complex_text,
                "query": "Analyze the sentiment of this text and explain your reasoning."
            })
            
            if not llm_sentiment or "error" in llm_sentiment:
                success = False
                failure_reason = f"Error in LLM sentiment analysis: {llm_sentiment.get('error', 'No result returned')}"
                logger.error(failure_reason)
            else:
                # Check for minimum content length as a basic quality check
                min_length = 100  # Expect at least 100 characters
                if len(llm_sentiment) < min_length:
                    success = False
                    failure_reason = f"LLM analysis too short: {len(llm_sentiment)} chars (expected {min_length})"
                    logger.error(failure_reason)
                else:
                    # Calculate a simple accuracy metric based on content relevance
                    # Check if the analysis mentions key sentiment terms
                    key_terms = ["positive", "negative", "neutral", "sentiment", "opinion", "mixed"]
                    term_matches = sum(1 for term in key_terms if term.lower() in llm_sentiment.lower())
                    accuracy = (term_matches / len(key_terms)) * 100
                    
                    logger.info(f"LLM sentiment analysis completed with {accuracy:.2f}% relevance score")
                    logger.info(f"Analysis excerpt: {llm_sentiment[:100]}...")
                
        except Exception as e:
            success = False
            failure_reason = f"LLM integration test failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "sentiment_llm_integration",
            success,
            duration,
            accuracy=accuracy,
            failure_reason=failure_reason
        )
        
    def run_tests(self):
        """Run all tests for this model"""
        super().run_tests()
        
        # Skip tests if model initialization failed
        if not self.model:
            self.results.add_warning("sentiment_tests", "Model not initialized, skipping tests")
            return
            
        # Prepare test data
        self.prepare_test_data()
        
        # Run tests
        self.verify_mcp_tools()
        self.test_text_sentiment()
        self.test_company_sentiment()
        self.test_llm_integration()
        
    def teardown(self):
        """Clean up after tests"""
        super().teardown()
        
        # Clean up Redis test data if needed
        if self.redis_mcp:
            for symbol in self.test_data.get("symbols", []):
                self.redis_mcp.call_tool(
                    "delete_key",
                    {"key": f"test:sentiment:{symbol}"}
                )


class MarketAnalysisModelTester(ModelAccuracyTester):
    """Test suite for the Market Analysis Model"""
    
    def __init__(self, results: AccuracyTestResults):
        super().__init__("Market Analysis Model", results)
        self.financial_data_mcp = None
        self.time_series_mcp = None
        self.redis_mcp = None
        
    def setup(self):
        """Set up the test environment"""
        super().setup()
        
        # Initialize required MCP tools
        self.financial_data_mcp = FinancialDataMCP()
        self.time_series_mcp = TimeSeriesMCP()
        self.redis_mcp = RedisMCP()
        
        # Test config with API keys from environment
        self.test_config = {
            "financial_data_config": {},
            "time_series_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": os.environ.get("LLM_MODEL", "anthropic/claude-3-opus-20240229"),
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
            "analysis_interval": 3600  # 1 hour for testing
        }
        
        # Initialize the model
        try:
            self.model = MarketAnalysisModel(config=self.test_config)
            logger.info("Market Analysis Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Market Analysis Model: {e}")
            self.results.record_test(
                "market_model_init",
                False,
                0.0,
                failure_reason=f"Initialization failed: {str(e)}"
            )
    
    def prepare_test_data(self):
        """Prepare test data for the model"""
        super().prepare_test_data()
        
        # Prepare test stocks
        self.test_data["symbols"] = TEST_STOCKS[:3]  # Use first 3 test stocks
        self.test_data["market_conditions"] = TEST_MARKET_CONDITIONS
        
        # Create expected results structure (will be populated during tests)
        self.expected_results = {
            "technical_indicators": {},
            "market_trends": {},
            "forecasts": {}
        }
    
    def verify_mcp_tools(self):
        """Verify that all required MCP tools are working"""
        super().verify_mcp_tools()
        
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            # Test financial data MCP
            symbol = self.test_data["symbols"][0]
            financial_data_result = self.financial_data_mcp.call_tool(
                "get_historical_data",
                {"symbol": symbol, "period": "1mo", "interval": "1d"}
            )
            
            if "error" in financial_data_result:
                success = False
                failure_reason = f"Financial Data MCP error: {financial_data_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info(f"Financial Data MCP successfully retrieved historical data for {symbol}")
            
            # Test time series MCP with sample data
            if "error" not in financial_data_result and "data" in financial_data_result:
                sample_data = financial_data_result["data"]
                
                # Calculate technical indicators
                indicators_result = self.time_series_mcp.call_tool(
                    "calculate_indicators",
                    {"data": sample_data, "indicators": ["SMA", "RSI"]}
                )
                
                if "error" in indicators_result:
                    success = False
                    failure_reason = f"Time Series MCP error: {indicators_result['error']}"
                    logger.error(failure_reason)
                else:
                    logger.info("Time Series MCP successfully calculated technical indicators")
            
            # Test Redis MCP
            test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
            redis_set_result = self.redis_mcp.call_tool(
                "set_json",
                {"key": "test:market", "value": test_data, "expiry": 60}
            )
            
            if "error" in redis_set_result:
                success = False
                failure_reason = f"Redis MCP set error: {redis_set_result['error']}"
                logger.error(failure_reason)
            else:
                # Try to get the data back
                redis_get_result = self.redis_mcp.call_tool(
                    "get_json",
                    {"key": "test:market"}
                )
                
                if "error" in redis_get_result:
                    success = False
                    failure_reason = f"Redis MCP get error: {redis_get_result['error']}"
                    logger.error(failure_reason)
                else:
                    logger.info("Redis MCP successfully stored and retrieved data")
                    
                # Clean up
                self.redis_mcp.call_tool(
                    "delete_key",
                    {"key": "test:market"}
                )
                
        except Exception as e:
            success = False
            failure_reason = f"MCP tool verification failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "market_mcp_tools",
            success,
            duration,
            failure_reason=failure_reason
        )
    
    def run_tests(self):
        """Run all tests for this model"""
        super().run_tests()
        
        # Skip tests if model initialization failed
        if not self.model:
            self.results.add_warning("market_tests", "Model not initialized, skipping tests")
            return
            
        # Prepare test data
        self.prepare_test_data()
        
        # Run tests
        self.verify_mcp_tools()
        # Add more specific tests for market analysis model


class RiskAssessmentModelTester(ModelAccuracyTester):
    """Test suite for the Risk Assessment Model"""
    
    def __init__(self, results: AccuracyTestResults):
        super().__init__("Risk Assessment Model", results)
        self.risk_analysis_mcp = None
        self.financial_data_mcp = None
        self.redis_mcp = None
        
    def setup(self):
        """Set up the test environment"""
        super().setup()
        
        # Initialize required MCP tools
        self.risk_analysis_mcp = RiskAnalysisMCP()
        self.financial_data_mcp = FinancialDataMCP()
        self.redis_mcp = RedisMCP()
        
        # Test config with API keys from environment
        self.test_config = {
            "risk_analysis_config": {},
            "financial_data_config": {},
            "redis_config": {},
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": os.environ.get("LLM_MODEL", "anthropic/claude-3-opus-20240229"),
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
            "analysis_interval": 3600  # 1 hour for testing
        }
        
        # Initialize the model
        try:
            self.model = RiskAssessmentModel(config=self.test_config)
            logger.info("Risk Assessment Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Risk Assessment Model: {e}")
            self.results.record_test(
                "risk_model_init",
                False,
                0.0,
                failure_reason=f"Initialization failed: {str(e)}"
            )
    
    def prepare_test_data(self):
        """Prepare test data for the model"""
        super().prepare_test_data()
        
        # Prepare test stocks
        self.test_data["symbols"] = TEST_STOCKS[:3]  # Use first 3 test stocks
        
        # Create expected results structure (will be populated during tests)
        self.expected_results = {
            "risk_metrics": {},
            "portfolio_risk": {},
            "risk_scenarios": {}
        }
    
    def verify_mcp_tools(self):
        """Verify that all required MCP tools are working"""
        super().verify_mcp_tools()
        
        start_time = time.time()
        success = True
        failure_reason = None
        
        try:
            # Test risk analysis MCP with sample data
            sample_returns = {
                "AAPL": [0.01, -0.005, 0.02, -0.01, 0.015],
                "MSFT": [0.008, -0.003, 0.015, -0.007, 0.01]
            }
            
            risk_metrics_result = self.risk_analysis_mcp.call_tool(
                "calculate_risk_metrics",
                {"returns_data": sample_returns}
            )
            
            if "error" in risk_metrics_result:
                success = False
                failure_reason = f"Risk Analysis MCP error: {risk_metrics_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info("Risk Analysis MCP successfully calculated risk metrics")
            
            # Test financial data MCP
            symbol = self.test_data["symbols"][0]
            financial_data_result = self.financial_data_mcp.call_tool(
                "get_historical_data",
                {"symbol": symbol, "period": "1mo", "interval": "1d"}
            )
            
            if "error" in financial_data_result:
                success = False
                failure_reason = f"Financial Data MCP error: {financial_data_result['error']}"
                logger.error(failure_reason)
            else:
                logger.info(f"Financial Data MCP successfully retrieved historical data for {symbol}")
            
            # Test Redis MCP
            test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
            redis_set_result = self.redis_mcp.call_tool(
                "set_json",
                {"key": "test:risk", "value": test_data, "expiry": 60}
            )
            
            if "error" in redis_set_result:
                success = False
                failure_reason = f"Redis MCP set error: {redis_set_result['error']}"
                logger.error(failure_reason)
            else:
                # Try to get the data back
                redis_get_result = self.redis_mcp.call_tool(
                    "get_json",
                    {"key": "test:risk"}
                )
                
                if "error" in redis_get_result:
                    success = False
                    failure_reason = f"Redis MCP get error: {redis_get_result['error']}"
                    logger.error(failure_reason)
                else:
                    logger.info("Redis MCP successfully stored and retrieved data")
                    
                # Clean up
                self.redis_mcp.call_tool(
                    "delete_key",
                    {"key": "test:risk"}
                )
                
        except Exception as e:
            success = False
            failure_reason = f"MCP tool verification failed: {str(e)}"
            logger.error(failure_reason, exc_info=True)
            
        duration = time.time() - start_time
        self.results.record_test(
            "risk_mcp_tools",
            success,
            duration,
            failure_reason=failure_reason
        )
    
    def run_tests(self):
        """Run all tests for this model"""
        super().run_tests()
        
        # Skip tests if model initialization failed
        if not self.model:
            self.results.add_warning("risk_tests", "Model not initialized, skipping tests")
            return
            
        # Prepare test data
        self.prepare_test_data()
        
        # Run tests
        self.verify_mcp_tools()
        # Add more specific tests for risk assessment model


# Map model names to tester classes
MODEL_TESTERS = {
    "fundamental": FundamentalAnalysisModelTester,
    "sentiment": SentimentAnalysisModelTester,
    "market": MarketAnalysisModelTester,
    "risk": RiskAssessmentModelTester,
    # Add other models here
}


def run_model_accuracy_tests(model_name: str) -> AccuracyTestResults:
    """
    Run accuracy tests for a specific model
    
    Args:
        model_name: Name of the model to test
        
    Returns:
        Test results object
    """
    results = AccuracyTestResults()
    
    if model_name not in MODEL_TESTERS:
        results.add_warning(f"{model_name}_tests", f"No tester class defined for {model_name}")
        return results
    
    logger.info(f"Running accuracy tests for {model_name}")
    
    # Create and run the tester
    tester_class = MODEL_TESTERS[model_name]
    tester = tester_class(results)
    
    try:
        # Setup
        tester.setup()
        
        # Run tests
        tester.run_tests()
        
        # Teardown
        tester.teardown()
        
    except Exception as e:
        logger.error(f"Error running tests for {model_name}: {e}", exc_info=True)
        results.record_test(
            f"{model_name}_tests",
            False,
            0.0,
            failure_reason=f"Unhandled exception: {str(e)}"
        )
    
    return results


def check_environment() -> bool:
    """
    Check if the environment is properly set up for testing
    
    Returns:
        True if environment is properly set up, False otherwise
    """
    print("\nChecking environment setup...")
    
    # Check for required API keys
    required_keys = {
        "OPENROUTER_API_KEY": "OpenRouter API (for LLM access)",
        "POLYGON_API_KEY": "Polygon API (for market data)",
        "YAHOO_FINANCE_API_KEY": "Yahoo Finance API (optional)",
        "REDIS_HOST": "Redis host (for database access)",
        "REDIS_PORT": "Redis port"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.environ.get(key):
            missing_keys.append(f"{key} ({description})")
            print(f"  ✗ {key} not found")
        else:
            print(f"  ✓ {key} found")
    
    if missing_keys:
        print("\nMissing environment variables. Please add them to your .env file:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nSome tests may fail without these variables.")
    else:
        print("\nAll required environment variables found.")
    
    # Check for required Python packages
    required_packages = [
        "dotenv", "numpy", "pandas", "torch", "transformers", "redis", "yfinance"
    ]
    
    missing_packages = []
    print("\nChecking for required Python packages...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} not installed")
    
    if missing_packages:
        print("\nMissing required packages. Please install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nSome tests may fail without these packages.")
    else:
        print("\nAll required Python packages found.")
    
    return len(missing_keys) == 0 and len(missing_packages) == 0


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run NextGen model accuracy tests")
    parser.add_argument("models", nargs="*", default=["all"], 
                      help="Name of models to test (default: all)")
    parser.add_argument("--check-env", action="store_true",
                      help="Check environment setup and exit")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Enable verbose output")
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*80)
    print(f"NEXTGEN MODELS ACCURACY TEST SUITE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check environment if requested
    if args.check_env:
        env_ok = check_environment()
        if not env_ok:
            print("\nEnvironment check failed. Some tests may not work correctly.")
            return
    
    # Determine which models to test
    models_to_test = []
    if "all" in args.models:
        models_to_test = list(MODEL_TESTERS.keys())
    else:
        for model in args.models:
            if model in MODEL_TESTERS:
                models_to_test.append(model)
            else:
                print(f"Unknown model: {model}")
                print(f"Available models: {', '.join(MODEL_TESTERS.keys())}")
                return
    
    # Print test plan
    print(f"\nRunning accuracy tests for: {', '.join(models_to_test)}\n")
    
    # Run tests for each model
    all_results = AccuracyTestResults()
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        results = run_model_accuracy_tests(model_name)
        
        # Merge results
        all_results.tests_run += results.tests_run
        all_results.tests_passed += results.tests_passed
        all_results.tests_failed += results.tests_failed
        all_results.failures.extend(results.failures)
        all_results.warnings.extend(results.warnings)
        
        # Merge performance metrics
        for test, durations in results.performance_metrics.items():
            if test not in all_results.performance_metrics:
                all_results.performance_metrics[test] = []
            all_results.performance_metrics[test].extend(durations)
        
        # Merge accuracy metrics
        for test, accuracies in results.accuracy_metrics.items():
            if test not in all_results.accuracy_metrics:
                all_results.accuracy_metrics[test] = []
            all_results.accuracy_metrics[test].extend(accuracies)
    
    # Print overall results
    all_results.print_summary()
    
    # Return exit code based on test results
    status = all_results.get_overall_status()
    print(f"\nOverall status: {status}")
    
    if status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
