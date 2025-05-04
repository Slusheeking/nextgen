"""
Analysis MCP Tools

This package contains specialized MCP tools for market analysis:
- Peak Detection: Identifies price peaks, valleys, and breakout patterns
- Slippage Analysis: Measures execution quality and timing
- Drift Detection: Identifies price drift from moving averages and trends
"""

from mcp_tools.analysis_mcp.peak_detection_mcp import PeakDetectionMCP
from mcp_tools.analysis_mcp.slippage_analysis_mcp import SlippageAnalysisMCP
from mcp_tools.analysis_mcp.drift_detection_mcp import DriftDetectionMCP

__all__ = [
    'PeakDetectionMCP',
    'SlippageAnalysisMCP',
    'DriftDetectionMCP'
]