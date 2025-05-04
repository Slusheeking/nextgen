"""
Analysis MCP Tools

This package contains specialized MCP tools for market analysis:
- Peak Detection: Identifies price peaks, valleys, and breakout patterns
- Slippage Analysis: Measures execution quality and timing
- Drift Detection: Identifies price drift from moving averages and trends
- Technical Indicators: Calculates technical indicators and patterns
- Growth Analysis: Analyzes growth metrics and compound rates
- Correlation Analysis: Measures correlations between assets
- Risk Metrics: Calculates VaR, volatility, and other risk metrics
- Portfolio Optimization: Optimizes portfolio allocations
"""

# Use lazy imports to avoid circular dependencies
# Import specific classes when needed

__version__ = "0.1.0"

__all__ = ["__version__"]

# Function to lazily import analysis MCP classes
def get_analysis_mcp(class_name):
    """
    Get an analysis MCP class lazily to avoid circular imports.
    
    Args:
        class_name (str): Name of the analysis MCP class
        
    Returns:
        class: The requested MCP class
        
    Raises:
        ImportError: If the class doesn't exist
    """
    class_map = {
        "PeakDetectionMCP": ".peak_detection_mcp",
        "SlippageAnalysisMCP": ".slippage_analysis_mcp",
        "DriftDetectionMCP": ".drift_detection_mcp",
        "TechnicalIndicatorsMCP": ".technical_indicators_mcp",
        "GrowthAnalysisMCP": ".growth_analysis_mcp",
        "CorrelationAnalysisMCP": ".correlation_analysis_mcp",
        "RiskMetricsMCP": ".risk_metrics_mcp",
        "PortfolioOptimizationMCP": ".portfolio_optimization_mcp",
        "FundamentalScoringMCP": ".fundamental_scoring_mcp",
        "PatternRecognitionMCP": ".pattern_recognition_mcp",
        "SentimentScoringMCP": ".sentiment_scoring_mcp",
        "EmbeddingsMCP": ".embeddings_mcp",
        "VectorDbMCP": ".vector_db_mcp",
        "DecisionAnalyticsMCP": ".decision_analytics_mcp",
    }
    
    if class_name not in class_map:
        raise ImportError(f"Unknown analysis MCP class: {class_name}")
        
    module_name = class_map[class_name]
    import importlib
    module = importlib.import_module(module_name, package="mcp_tools.analysis_mcp")
    return getattr(module, class_name)
