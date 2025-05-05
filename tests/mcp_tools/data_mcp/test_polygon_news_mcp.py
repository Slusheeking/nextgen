#!/usr/bin/env python3
"""
Test suite for the Polygon News MCP Tool.

This module tests the functionality of the PolygonNewsMCP class, including:
- Initialization and configuration
- News data processing
- Endpoint handling
- Public API methods
- Sentiment analysis
- Error handling and retry mechanisms
- Data throughput integrity
"""

import os
import unittest
import json
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

# Create proper mocks for setup_monitoring
mock_monitor = MagicMock()
mock_metrics = MagicMock()
mock_setup_monitoring = MagicMock(return_value=(mock_monitor, mock_metrics))

# Mock all required modules in correct order
sys.modules['utils'] = MagicMock()
sys.modules['utils.env_loader'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.sentiment'] = MagicMock()
sys.modules['nltk.sentiment.vader'] = MagicMock()

# Set up monitoring mock chain
# Set up monitoring mock chain
sys.modules['monitoring'] = MagicMock()
sys.modules['monitoring.prometheus_loki_utils'] = MagicMock()
sys.modules['monitoring.prometheus_loki_utils'].setup_monitoring = mock_setup_monitoring
sys.modules['monitoring.loki'] = MagicMock()
sys.modules['monitoring.loki.loki_manager'] = MagicMock()

# Add mocks for monitoring submodules to fix import errors
sys.modules['monitoring.netdata_logger'] = MagicMock()
sys.modules['monitoring.system_metrics'] = MagicMock()

# Now import the module with proper patch
with patch('mcp_tools.base_mcp_server.BaseMCPServer'), \
     patch('monitoring.prometheus_loki_utils.setup_monitoring', mock_setup_monitoring), \
     patch('monitoring.setup_monitoring', mock_setup_monitoring):
    from mcp_tools.data_mcp.polygon_news_mcp import PolygonNewsMCP