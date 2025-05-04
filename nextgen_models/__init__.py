"""
FinGPT AI Day Trading System

A comprehensive, AI-powered day trading system built on the FinGPT framework,
using large language models to analyze financial markets and execute trades.
"""

__version__ = "1.0.0"

# Use lazy imports to avoid circular dependencies
# Import components when needed directly:
# from nextgen_models.autogen_orchestrator.autogen_model import AutoGenOrchestrator

__all__ = ["__version__"]

# Function to lazily import model classes
def get_model_class(class_name):
    """
    Lazily import and return a model class to avoid circular dependencies.
    
    Args:
        class_name (str): Name of the model class to import
        
    Returns:
        class: The imported class
    
    Raises:
        ImportError: If the class doesn't exist
    """
    class_map = {
        "AutoGenOrchestrator": ".autogen_orchestrator.autogen_model",
        "SelectionModel": ".nextgen_select.select_model",
        "SentimentAnalysisModel": ".nextgen_sentiment_analysis.sentiment_analysis_model",
        "MarketAnalysisModel": ".nextgen_market_analysis.market_analysis_model",
        "RiskAssessmentModel": ".nextgen_risk_assessment.risk_assessment_model",
        "TradeModel": ".nextgen_trader.trade_model",
        "FundamentalAnalysisModel": ".nextgen_fundamental_analysis.fundamental_analysis_model",
        "DecisionModel": ".nextgen_decision.decision_model",
        "ContextModel": ".nextgen_context_model.context_model",
    }
    
    if class_name not in class_map:
        raise ImportError(f"Unknown model class: {class_name}")
    
    module_name = class_map[class_name]
    import importlib
    module = importlib.import_module(module_name, package="nextgen_models")
    return getattr(module, class_name)
