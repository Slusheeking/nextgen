"""
Test script for validating pyautogen 0.9.0 (AG2) integration and MCP tools functionality
in the NextGen FinGPT system.

This test focuses on:
1. Validating that the updated code successfully uses pyautogen 0.9.0 (AG2) patterns
   - Test agent creation with AG2 patterns
   - Verify function registration using the register_function decorator
   - Test that properly annotated functions are correctly exposed to agents

2. Testing the MCP tools integration
   - Verify that MCP tool calls work correctly through AG2 function registration
   - Test that the MCP client interface is properly exposed
   - Validate error handling for MCP tool calls
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
from typing import Dict, List, Any, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from fingpt.autogen_orchestrator.autogen_model import AutoGenOrchestrator
from fingpt.nextgen_selection.selection_model import SelectionModel
import autogen
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function
    # FunctionCall is not available in autogen 0.9.0
)
from mcp_tools.base_mcp_server import BaseMCPServer


class TestAG2Integration(unittest.TestCase):
    """Test suite for validating AG2 integration and MCP tools functionality."""

    def setUp(self):
        """Set up test environment with mocked dependencies."""
        # Mock environment variables and configurations
        self.env_patcher = patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-api-key',
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_DB': '0',
            'POLYGON_API_KEY': 'test-polygon-key',
            'ALPACA_API_KEY': 'test-alpaca-key',
            'ALPACA_API_SECRET': 'test-alpaca-secret'
        })
        self.env_patcher.start()
        
        # Create a minimal test configuration
        self.test_config = {
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": "test-api-key",
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None
                    }
                ]
            }
        }
        
        # Create mock MCP servers
        self.mock_redis_mcp = self._create_mock_mcp_server("redis_mcp")
        self.mock_alpaca_mcp = self._create_mock_mcp_server("alpaca_mcp")
        self.mock_polygon_rest_mcp = self._create_mock_mcp_server("polygon_rest_mcp")
        self.mock_polygon_ws_mcp = self._create_mock_mcp_server("polygon_ws_mcp")
        self.mock_unusual_whales_mcp = self._create_mock_mcp_server("unusual_whales_mcp")
        
        # Set up patches for MCP servers
        self.redis_patcher = patch('fingpt.nextgen_selection.selection_model.RedisMCP', return_value=self.mock_redis_mcp)
        self.alpaca_patcher = patch('fingpt.nextgen_selection.selection_model.AlpacaMCP', return_value=self.mock_alpaca_mcp)
        self.polygon_rest_patcher = patch('fingpt.nextgen_selection.selection_model.PolygonRestMCP', return_value=self.mock_polygon_rest_mcp)
        self.polygon_ws_patcher = patch('fingpt.nextgen_selection.selection_model.PolygonWSMCP', return_value=self.mock_polygon_ws_mcp)
        self.unusual_whales_patcher = patch('fingpt.nextgen_selection.selection_model.UnusualWhalesMCP', return_value=self.mock_unusual_whales_mcp)
        
        # Start all patches
        self.redis_mock = self.redis_patcher.start()
        self.alpaca_mock = self.alpaca_patcher.start()
        self.polygon_rest_mock = self.polygon_rest_patcher.start()
        self.polygon_ws_mock = self.polygon_ws_patcher.start()
        self.unusual_whales_mock = self.unusual_whales_patcher.start()
        
        # Mock autogen's LLM calls to avoid actual API calls
        self.autogen_patcher = patch('autogen.ChatCompletion.create', return_value={"choices": [{"message": {"content": "Test response"}}]})
        self.autogen_mock = self.autogen_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop all patches
        self.env_patcher.stop()
        self.redis_patcher.stop()
        self.alpaca_patcher.stop()
        self.polygon_rest_patcher.stop()
        self.polygon_ws_patcher.stop()
        self.unusual_whales_patcher.stop()
        self.autogen_patcher.stop()
    
    def _create_mock_mcp_server(self, name):
        """Create a mock MCP server with basic functionality."""
        mock_server = MagicMock(spec=BaseMCPServer)
        mock_server.name = name
        mock_server.tools = {
            "test_tool": {
                "func": lambda **kwargs: {"result": "success", **kwargs},
                "description": "Test tool for testing"
            },
            "error_tool": {
                "func": lambda **kwargs: {"error": "Test error"},
                "description": "Tool that always returns an error"
            }
        }
        
        # Set up call_tool method to route to the appropriate mock tool
        def mock_call_tool(tool_name, args=None):
            args = args or {}
            if tool_name not in mock_server.tools:
                raise ValueError(f"Tool not found: {tool_name}")
            return mock_server.tools[tool_name]["func"](**args)
        
        mock_server.call_tool = mock_call_tool
        
        # Set up list_tools method
        mock_server.list_tools.return_value = [
            {"name": "test_tool", "description": "Test tool for testing"},
            {"name": "error_tool", "description": "Tool that always returns an error"}
        ]
        
        return mock_server

    def test_agent_creation_with_ag2(self):
        """Test that agents are created correctly using AG2 patterns."""
        # Initialize the selection model with mocked dependencies
        selection_model = SelectionModel(self.test_config)
        
        # Verify that agents were created with AG2 patterns
        self.assertIsInstance(selection_model.agents["selection_assistant"], AssistantAgent)
        self.assertIsInstance(selection_model.agents["data_assistant"], AssistantAgent)
        self.assertIsInstance(selection_model.agents["user_proxy"], UserProxyAgent)
        
        # Verify that agents have the expected properties
        self.assertEqual(selection_model.agents["selection_assistant"].name, "StockSelectionAssistant")
        self.assertEqual(selection_model.agents["data_assistant"].name, "DataAnalysisAssistant")
        
        # Verify that the user proxy is configured correctly
        user_proxy = selection_model.agents["user_proxy"]
        self.assertEqual(user_proxy.human_input_mode, "NEVER")
        self.assertEqual(user_proxy.max_consecutive_auto_reply, 10)
        
        # Test orchestrator agent creation
        orchestrator = AutoGenOrchestrator()
        
        # Verify that orchestrator agents were created with AG2 patterns
        self.assertIn("selection", orchestrator.agents)
        self.assertIn("data", orchestrator.agents)
        self.assertIn("finnlp", orchestrator.agents)
        self.assertIn("forecaster", orchestrator.agents)
        self.assertIn("rag", orchestrator.agents)
        self.assertIn("execution", orchestrator.agents)
        self.assertIn("monitoring", orchestrator.agents)
        self.assertIn("human_proxy", orchestrator.agents)
        
        # Verify that the group chat is set up correctly
        self.assertIsNotNone(orchestrator.group_chat)
        self.assertIsNotNone(orchestrator.chat_manager)
        self.assertEqual(orchestrator.group_chat.max_round, 10)
        self.assertEqual(orchestrator.group_chat.speaker_selection_method, "round_robin")

    def test_function_registration_with_ag2(self):
        """Test that functions are registered correctly using AG2 patterns."""
        # Initialize the selection model with mocked dependencies
        selection_model = SelectionModel(self.test_config)
        
        # Get the user proxy agent
        user_proxy = selection_model.agents["user_proxy"]
        
        # Verify that functions were registered with the user proxy
        registered_functions = user_proxy._function_map
        
        # Check for core selection functions
        self.assertIn("get_market_data", registered_functions)
        self.assertIn("get_technical_indicators", registered_functions)
        self.assertIn("get_unusual_activity", registered_functions)
        self.assertIn("filter_by_liquidity", registered_functions)
        self.assertIn("score_candidates", registered_functions)
        
        # Check for storage and retrieval functions
        self.assertIn("store_candidates", registered_functions)
        self.assertIn("get_candidates", registered_functions)
        self.assertIn("get_top_candidates", registered_functions)
        self.assertIn("get_candidate", registered_functions)
        self.assertIn("get_market_context", registered_functions)
        
        # Check for MCP tool access functions
        self.assertIn("use_alpaca_tool", registered_functions)
        self.assertIn("use_redis_tool", registered_functions)
        self.assertIn("use_polygon_rest_tool", registered_functions)
        self.assertIn("use_polygon_ws_tool", registered_functions)
        self.assertIn("use_unusual_whales_tool", registered_functions)
        self.assertIn("list_mcp_tools", registered_functions)
        
        # Test orchestrator function registration
        orchestrator = AutoGenOrchestrator()
        orchestrator.connect_to_models({"selection_config": self.test_config})
        
        # Get the user proxy agent
        orchestrator_user_proxy = orchestrator.agents["human_proxy"]
        
        # Verify that functions were registered with the orchestrator's user proxy
        orchestrator_functions = orchestrator_user_proxy._function_map
        
        # Check for selection model functions
        self.assertIn("run_selection_cycle", orchestrator_functions)
        self.assertIn("get_selection_data", orchestrator_functions)
        self.assertIn("get_market_data", orchestrator_functions)
        self.assertIn("get_technical_indicators", orchestrator_functions)
        
        # Check for MCP tool access functions
        self.assertIn("use_mcp_tool", orchestrator_functions)
        self.assertIn("list_mcp_tools", orchestrator_functions)

    def test_function_calling_with_ag2(self):
        """Test that functions can be called correctly using AG2 patterns."""
        # Initialize the selection model with mocked dependencies
        selection_model = SelectionModel(self.test_config)
        
        # Mock the get_candidates method to return test data
        test_candidates = [{"ticker": "AAPL", "score": 95}, {"ticker": "MSFT", "score": 90}]
        selection_model.get_candidates = MagicMock(return_value=test_candidates)
        
        # Get the user proxy agent
        user_proxy = selection_model.agents["user_proxy"]
        
        # Call the function through the user proxy
        result = user_proxy.get_candidates()
        
        # Verify that the function was called and returned the expected result
        selection_model.get_candidates.assert_called_once()
        self.assertEqual(result, test_candidates)
        
        # Test function with parameters
        selection_model.get_top_candidates = MagicMock(return_value=test_candidates[:1])
        result = user_proxy.get_top_candidates(limit=1)
        selection_model.get_top_candidates.assert_called_once_with(limit=1)
        self.assertEqual(result, test_candidates[:1])

    def test_mcp_tool_registration(self):
        """Test that MCP tools are registered correctly."""
        # Initialize the selection model with mocked dependencies
        selection_model = SelectionModel(self.test_config)
        
        # Verify that MCP servers were initialized
        self.assertIsNotNone(selection_model.alpaca_mcp)
        self.assertIsNotNone(selection_model.redis_mcp)
        self.assertIsNotNone(selection_model.polygon_rest_mcp)
        self.assertIsNotNone(selection_model.polygon_ws_mcp)
        self.assertIsNotNone(selection_model.unusual_whales_mcp)
        
        # Get the user proxy agent
        user_proxy = selection_model.agents["user_proxy"]
        
        # Verify that MCP tool access functions were registered
        registered_functions = user_proxy._function_map
        self.assertIn("use_alpaca_tool", registered_functions)
        self.assertIn("use_redis_tool", registered_functions)
        self.assertIn("use_polygon_rest_tool", registered_functions)
        self.assertIn("use_polygon_ws_tool", registered_functions)
        self.assertIn("use_unusual_whales_tool", registered_functions)
        self.assertIn("list_mcp_tools", registered_functions)

    def test_mcp_tool_calling(self):
        """Test that MCP tools can be called correctly."""
        # Initialize the selection model with mocked dependencies
        selection_model = SelectionModel(self.test_config)
        
        # Get the user proxy agent
        user_proxy = selection_model.agents["user_proxy"]
        
        # Call the list_mcp_tools function
        result = user_proxy.list_mcp_tools(server_name="redis")
        
        # Verify that the function was called and returned the expected result
        self.mock_redis_mcp.list_tools.assert_called_once()
        self.assertEqual(result, self.mock_redis_mcp.list_tools.return_value)
        
        # Call a tool through the MCP server
        test_args = {"param1": "value1", "param2": "value2"}
        result = user_proxy.use_redis_tool(tool_name="test_tool", arguments=test_args)
        
        # Verify that the tool was called with the correct arguments
        self.mock_redis_mcp.call_tool.assert_called_with("test_tool", test_args)
        self.assertEqual(result, {"result": "success", **test_args})

    def test_mcp_tool_error_handling(self):
        """Test error handling for MCP tool calls."""
        # Initialize the selection model with mocked dependencies
        selection_model = SelectionModel(self.test_config)
        
        # Get the user proxy agent
        user_proxy = selection_model.agents["user_proxy"]
        
        # Call a tool that returns an error
        result = user_proxy.use_redis_tool(tool_name="error_tool", arguments={})
        
        # Verify that the error was handled correctly
        self.mock_redis_mcp.call_tool.assert_called_with("error_tool", {})
        self.assertEqual(result, {"error": "Test error"})
        
        # Call a tool that doesn't exist
        self.mock_redis_mcp.call_tool.side_effect = ValueError("Tool not found: nonexistent_tool")
        result = user_proxy.use_redis_tool(tool_name="nonexistent_tool", arguments={})
        
        # Verify that the exception was handled correctly
        self.assertIn("error", result)
        self.assertIn("Tool not found", result["error"])

    def test_orchestrator_mcp_integration(self):
        """Test that the orchestrator integrates correctly with MCP tools."""
        # Initialize the orchestrator with mocked dependencies
        orchestrator = AutoGenOrchestrator()
        orchestrator.connect_to_models({"selection_config": self.test_config})
        
        # Get the user proxy agent
        user_proxy = orchestrator.agents["human_proxy"]
        
        # Verify that MCP tool access functions were registered
        registered_functions = user_proxy._function_map
        self.assertIn("use_mcp_tool", registered_functions)
        self.assertIn("list_mcp_tools", registered_functions)
        
        # Mock the selection model's MCP servers
        orchestrator.models["selection"] = MagicMock()
        orchestrator.models["selection"].alpaca_mcp = self.mock_alpaca_mcp
        orchestrator.models["selection"].redis_mcp = self.mock_redis_mcp
        orchestrator.models["selection"].polygon_rest_mcp = self.mock_polygon_rest_mcp
        orchestrator.models["selection"].polygon_ws_mcp = self.mock_polygon_ws_mcp
        orchestrator.models["selection"].unusual_whales_mcp = self.mock_unusual_whales_mcp
        
        # Call the use_mcp_tool function
        test_args = {"param1": "value1", "param2": "value2"}
        result = user_proxy.use_mcp_tool(
            server_name="redis",
            tool_name="test_tool",
            arguments=test_args
        )
        
        # Verify that the tool was called with the correct arguments
        self.mock_redis_mcp.call_tool.assert_called_with("test_tool", test_args)
        self.assertEqual(result, {"result": "success", **test_args})
        
        # Test error handling for nonexistent server
        result = user_proxy.use_mcp_tool(
            server_name="nonexistent_server",
            tool_name="test_tool",
            arguments=test_args
        )
        
        # Verify that the error was handled correctly
        self.assertIn("error", result)
        self.assertIn("MCP server not found", result["error"])


class TestRegisterFunctionDecorator(unittest.TestCase):
    """Test suite for validating the register_function decorator."""
    
    def test_register_function_decorator(self):
        """Test that the register_function decorator works correctly."""
        # Create a user proxy agent
        user_proxy = UserProxyAgent(
            name="TestUserProxy",
            human_input_mode="NEVER"
        )
        
        # Define a function with the register_function decorator
        @register_function(
            name="test_function",
            description="A test function",
            parameters={
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter"
                }
            },
            return_type=Dict[str, Any]
        )
        def test_function(param1: str, param2: int) -> Dict[str, Any]:
            return {"param1": param1, "param2": param2}
        
        # Register the function with the user proxy
        user_proxy.register_function(test_function)
        
        # Verify that the function was registered correctly
        self.assertIn("test_function", user_proxy._function_map)
        
        # Call the function through the user proxy
        result = user_proxy.test_function(param1="test", param2=42)
        
        # Verify that the function was called and returned the expected result
        self.assertEqual(result, {"param1": "test", "param2": 42})


if __name__ == "__main__":
    unittest.main()