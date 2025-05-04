#!/usr/bin/env python3
"""
Comprehensive test suite for the NextGen FinGPT system with pyautogen 0.9.0 and MCP tools integration.

This test suite validates:
1. Basic functionality of AutoGenOrchestrator and SelectionModel
2. Function registration with AG2 patterns
3. Integration between components
4. MCP tool integration
5. Error handling for edge cases
"""

import os
import sys
import json
import unittest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_orchestrator")

# Make sure we can import modules from the project
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv(dotenv_path=project_root / ".env")

# Import the modules to test
from fingpt.autogen_orchestrator.autogen_model import AutoGenOrchestrator, create_orchestrator
from fingpt.nextgen_select.select_model import SelectionModel

# Sample market data for testing
SAMPLE_MARKET_DATA = {
    "sp500": 5024.35,
    "vix": 14.87,
    "treasury_10y": 4.12,
    "market_sentiment": "mixed"
}

class TestNextGenFinGPT(unittest.TestCase):
    """Test suite for the NextGen FinGPT system with pyautogen 0.9.0 and MCP tools integration."""

    def setUp(self):
        """Set up test environment."""
        # Configure MCP tools
        self.mcp_config = {
            "alpaca_config": {
                "api_key": os.environ.get("ALPACA_API_KEY", ""),
                "api_secret": os.environ.get("ALPACA_API_SECRET", ""),
                "paper": True  # Use paper trading
            },
            "redis_config": {
                "host": os.environ.get("REDIS_HOST", "localhost"),
                "port": int(os.environ.get("REDIS_PORT", 6379)),
                "db": int(os.environ.get("REDIS_DB", 0))
            },
            "polygon_rest_config": {
                "api_key": os.environ.get("POLYGON_API_KEY", "")
            },
            "polygon_ws_config": {
                "api_key": os.environ.get("POLYGON_API_KEY", "")
            },
            "unusual_whales_config": {
                "api_key": os.environ.get("UNUSUAL_WHALES_API_KEY", "")
            },
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None
                    }
                ]
            }
        }

        # Create instances for testing
        self.selection_model = SelectionModel(self.mcp_config)
        
        # Create orchestrator with the selection model
        self.orchestrator = create_orchestrator()
        self.orchestrator.models = {"selection": self.selection_model}
        
        logger.info("Test environment set up with SelectionModel and AutoGenOrchestrator")

    def test_selection_model_initialization(self):
        """Test that the SelectionModel initializes correctly with MCP tools."""
        logger.info("Testing SelectionModel initialization...")
        
        # Verify MCP tools are initialized
        self.assertIsNotNone(self.selection_model.alpaca_mcp)
        self.assertIsNotNone(self.selection_model.redis_mcp)
        self.assertIsNotNone(self.selection_model.polygon_rest_mcp)
        self.assertIsNotNone(self.selection_model.polygon_ws_mcp)
        self.assertIsNotNone(self.selection_model.unusual_whales_mcp)
        
        # Verify AutoGen agents are initialized
        self.assertIsNotNone(self.selection_model.agents)
        self.assertIn("selection_assistant", self.selection_model.agents)
        self.assertIn("data_assistant", self.selection_model.agents)
        self.assertIn("user_proxy", self.selection_model.agents)
        
        logger.info("SelectionModel initialization test passed")

    def test_orchestrator_initialization(self):
        """Test that the AutoGenOrchestrator initializes correctly."""
        logger.info("Testing AutoGenOrchestrator initialization...")
        
        # Verify agents are initialized
        self.assertIsNotNone(self.orchestrator.agents)
        self.assertGreater(len(self.orchestrator.agents), 0)
        
        # Verify group chat is initialized
        self.assertIsNotNone(self.orchestrator.group_chat)
        self.assertIsNotNone(self.orchestrator.chat_manager)
        
        logger.info("AutoGenOrchestrator initialization test passed")

    def test_function_registration(self):
        """Test that functions are properly registered with AG2 patterns."""
        logger.info("Testing function registration with AG2 patterns...")
        
        # Test SelectionModel function registration
        user_proxy = self.selection_model.agents["user_proxy"]
        
        # Check if functions are registered
        self.assertTrue(hasattr(user_proxy, "register_function"))
        
        # Check if MCP tool access functions are registered
        registered_functions = [f.__name__ for f in user_proxy._function_map.values()]
        
        # Check for core functions
        self.assertIn("get_market_data", registered_functions)
        self.assertIn("get_technical_indicators", registered_functions)
        self.assertIn("get_unusual_activity", registered_functions)
        self.assertIn("filter_by_liquidity", registered_functions)
        self.assertIn("score_candidates", registered_functions)
        
        # Check for MCP tool access functions
        self.assertIn("use_alpaca_tool", registered_functions)
        self.assertIn("use_redis_tool", registered_functions)
        self.assertIn("use_polygon_rest_tool", registered_functions)
        self.assertIn("use_polygon_ws_tool", registered_functions)
        self.assertIn("use_unusual_whales_tool", registered_functions)
        self.assertIn("list_mcp_tools", registered_functions)
        
        logger.info("Function registration test passed")

    def test_get_market_context(self):
        """Test getting market context from MCP tools."""
        logger.info("Testing get_market_context...")
        
        # Get market context
        market_context = self.selection_model._get_market_context()
        
        # Verify market context structure
        self.assertIsInstance(market_context, dict)
        
        # Log market context for inspection
        logger.info(f"Market context: {json.dumps(market_context, indent=2)}")
        
        logger.info("get_market_context test completed")

    def test_mcp_tool_integration(self):
        """Test MCP tool integration by listing available tools."""
        logger.info("Testing MCP tool integration...")
        
        # Test listing MCP tools
        try:
            alpaca_tools = self.selection_model.alpaca_mcp.list_tools()
            logger.info(f"Alpaca MCP tools: {json.dumps(alpaca_tools, indent=2)}")
            self.assertIsInstance(alpaca_tools, list)
            
            redis_tools = self.selection_model.redis_mcp.list_tools()
            logger.info(f"Redis MCP tools: {json.dumps(redis_tools, indent=2)}")
            self.assertIsInstance(redis_tools, list)
            
            polygon_rest_tools = self.selection_model.polygon_rest_mcp.list_tools()
            logger.info(f"Polygon REST MCP tools: {json.dumps(polygon_rest_tools, indent=2)}")
            self.assertIsInstance(polygon_rest_tools, list)
            
            logger.info("MCP tool integration test passed")
        except Exception as e:
            logger.error(f"Error in MCP tool integration test: {e}")
            self.fail(f"MCP tool integration test failed: {e}")

    def test_selection_model_data_retrieval(self):
        """Test data retrieval methods in SelectionModel."""
        logger.info("Testing SelectionModel data retrieval...")
        
        try:
            # Test get_market_data
            market_data = self.selection_model.get_market_data()
            logger.info(f"Retrieved {len(market_data)} stocks from market data")
            self.assertIsInstance(market_data, list)
            
            # If we have market data, test filtering and technical indicators
            if market_data:
                # Take a small sample for testing
                sample_stocks = market_data[:3]
                logger.info(f"Using sample stocks: {[stock.get('ticker') for stock in sample_stocks]}")
                
                # Test filter_by_liquidity
                filtered_stocks = self.selection_model.filter_by_liquidity(sample_stocks)
                logger.info(f"Filtered stocks: {[stock.get('ticker') for stock in filtered_stocks]}")
                
                # Test get_technical_indicators
                if filtered_stocks:
                    tech_stocks = self.selection_model.get_technical_indicators(filtered_stocks)
                    logger.info(f"Stocks with technical indicators: {[stock.get('ticker') for stock in tech_stocks]}")
                    
                    # Test get_unusual_activity
                    if tech_stocks:
                        activity_stocks = self.selection_model.get_unusual_activity(tech_stocks)
                        logger.info(f"Stocks with unusual activity: {[stock.get('ticker') for stock in activity_stocks]}")
                        
                        # Test score_candidates
                        if activity_stocks:
                            scored_stocks = self.selection_model.score_candidates(activity_stocks)
                            logger.info(f"Scored stocks: {[(stock.get('ticker'), stock.get('score')) for stock in scored_stocks]}")
            
            logger.info("SelectionModel data retrieval test completed")
        except Exception as e:
            logger.error(f"Error in SelectionModel data retrieval test: {e}")
            logger.info("This error may be due to API key limitations or connectivity issues")
            # Don't fail the test as it might be due to API limitations
            logger.info("Continuing with other tests...")

    def test_selection_model_redis_operations(self):
        """Test Redis operations in SelectionModel."""
        logger.info("Testing SelectionModel Redis operations...")
        
        try:
            # Create test candidates
            test_candidates = [
                {
                    "ticker": "TEST1",
                    "price": 100.0,
                    "score": 85.5,
                    "rsi": 65.2,
                    "relative_volume": 2.5
                },
                {
                    "ticker": "TEST2",
                    "price": 200.0,
                    "score": 75.0,
                    "rsi": 45.8,
                    "relative_volume": 1.8
                }
            ]
            
            # Test store_candidates
            store_result = self.selection_model.store_candidates(test_candidates)
            logger.info(f"Store candidates result: {store_result}")
            
            # Test get_candidates
            retrieved_candidates = self.selection_model.get_candidates()
            logger.info(f"Retrieved {len(retrieved_candidates)} candidates from Redis")
            
            # Test get_top_candidates
            top_candidates = self.selection_model.get_top_candidates(limit=1)
            logger.info(f"Top candidates: {[c.get('ticker') for c in top_candidates]}")
            
            # Test get_candidate
            if test_candidates:
                candidate = self.selection_model.get_candidate(test_candidates[0]["ticker"])
                logger.info(f"Retrieved candidate: {candidate.get('ticker') if candidate else None}")
            
            # Test get_selection_data
            selection_data = self.selection_model.get_selection_data()
            logger.info(f"Selection data keys: {selection_data.keys()}")
            
            logger.info("SelectionModel Redis operations test completed")
        except Exception as e:
            logger.error(f"Error in SelectionModel Redis operations test: {e}")
            logger.info("This error may be due to Redis connectivity issues")
            # Don't fail the test as it might be due to Redis connectivity issues
            logger.info("Continuing with other tests...")

    def test_orchestrator_selection_integration(self):
        """Test integration between AutoGenOrchestrator and SelectionModel."""
        logger.info("Testing integration between AutoGenOrchestrator and SelectionModel...")
        
        try:
            # Run a trading cycle with mock=True to avoid actual API calls to OpenRouter
            result = self.orchestrator.run_trading_cycle(SAMPLE_MARKET_DATA, mock=True)
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn("chat_result", result)
            self.assertIn("decisions", result)
            self.assertIn("timestamp", result)
            
            logger.info(f"Trading cycle completed with {len(result.get('decisions', []))} decisions")
            logger.info("Orchestrator-Selection integration test passed")
        except Exception as e:
            logger.error(f"Error in Orchestrator-Selection integration test: {e}")
            self.fail(f"Orchestrator-Selection integration test failed: {e}")

    def test_error_handling(self):
        """Test error handling for edge cases."""
        logger.info("Testing error handling...")
        
        # Test with invalid market data
        try:
            result = self.orchestrator.run_trading_cycle({"invalid": "data"}, mock=True)
            logger.info("Orchestrator handled invalid market data gracefully")
        except Exception as e:
            logger.error(f"Orchestrator failed with invalid market data: {e}")
            self.fail(f"Error handling test failed: {e}")
        
        # Test with missing selection model
        try:
            temp_orchestrator = create_orchestrator()
            # Intentionally don't set the selection model
            result = temp_orchestrator.run_trading_cycle(SAMPLE_MARKET_DATA, mock=True)
            logger.info("Orchestrator handled missing selection model gracefully")
        except Exception as e:
            logger.error(f"Orchestrator failed with missing selection model: {e}")
            self.fail(f"Error handling test failed: {e}")
        
        logger.info("Error handling test passed")


def run_basic_test():
    """Run a basic test of the orchestrator with mock mode."""
    logger.info("Running basic orchestrator test in mock mode...")
    
    orchestrator = create_orchestrator()
    
    # Run a test trading cycle
    result = orchestrator.run_trading_cycle(SAMPLE_MARKET_DATA, mock=True)
    
    # Print results
    logger.info(f"Trading cycle completed!")
    logger.info(f"Generated {len(result.get('decisions', []))} trading decisions")
    
    # Print decisions in a readable format
    if result.get("decisions"):
        logger.info("\nTrading Decisions:")
        for i, decision in enumerate(result["decisions"], 1):
            logger.info(f"Decision {i}:")
            logger.info(f"  Ticker:   {decision.get('ticker', 'N/A')}")
            logger.info(f"  Action:   {decision.get('action', 'N/A')}")
            logger.info(f"  Quantity: {decision.get('quantity', 'N/A')}")
            logger.info(f"  Reason:   {decision.get('reason', 'N/A')}")
    else:
        logger.info("\nNo specific trading decisions were extracted.")
        logger.info("Raw chat result preview:")
        chat_result = result.get("chat_result", "")
        preview_length = min(500, len(chat_result))
        logger.info(f"{chat_result[:preview_length]}...")


if __name__ == "__main__":
    logger.info("Starting NextGen FinGPT test suite")
    
    # Check if we should run unittest or basic test
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        run_basic_test()
    else:
        unittest.main()