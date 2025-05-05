"""
Integration Example - Using NetdataLogger with NextGen Components

This file demonstrates how to integrate NetdataLogger with existing
NextGen components like the trade model, risk assessment, etc.
"""

import os
import sys
import time
import random

# Add project root to path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import NetdataLogger
from monitoring.netdata_logger import NetdataLogger

# Import NextGen components
from nextgen_models.nextgen_trader.trade_model import TradeModel
from nextgen_models.nextgen_risk_assessment.risk_assessment_model import RiskAssessmentModel
from nextgen_models.nextgen_market_analysis.market_analysis_model import MarketAnalysisModel


class MonitoredTradeModel:
    """
    Example wrapper for TradeModel that adds monitoring with NetdataLogger
    """
    
    def __init__(self, config_path=None):
        # Initialize the logger
        self.logger = NetdataLogger(component_name="trade-model")
        self.logger.info("Initializing monitored trade model")
        
        # Initialize the actual trade model
        try:
            self.model = TradeModel(config_path)
            self.logger.info("Trade model initialized successfully", config_path=config_path)
        except Exception as e:
            self.logger.error("Failed to initialize trade model", error=str(e), config_path=config_path)
            raise
        
        # Start system metrics collection
        self.logger.start_system_metrics(interval=10)
    
    def execute_trade(self, symbol, quantity, price, side):
        """Execute a trade with monitoring"""
        self.logger.info("Executing trade", 
                        symbol=symbol, 
                        quantity=quantity, 
                        price=price, 
                        side=side)
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Execute the trade using the actual model
            result = self.model.execute_trade(symbol, quantity, price, side)
            
            # Calculate execution time
            duration = time.time() - start_time
            
            # Record metrics
            self.logger.timing("trade_execution_time_ms", duration * 1000)
            self.logger.counter("trades_executed")
            self.logger.gauge("trade_value", quantity * price)
            
            # Log success
            self.logger.info("Trade executed successfully", 
                           trade_id=result.get('trade_id', 'unknown'),
                           execution_time_ms=duration * 1000)
            
            return result
            
        except Exception as e:
            # Calculate execution time even for failures
            duration = time.time() - start_time
            
            # Record error metrics
            self.logger.counter("trade_errors")
            self.logger.timing("trade_error_time_ms", duration * 1000)
            
            # Log error
            self.logger.error("Trade execution failed", 
                             symbol=symbol, 
                             quantity=quantity, 
                             price=price, 
                             side=side,
                             error=str(e),
                             execution_time_ms=duration * 1000)
            raise
    
    def analyze_market(self, symbols, timeframe):
        """Analyze market with monitoring"""
        self.logger.info("Analyzing market", 
                        symbols=symbols, 
                        timeframe=timeframe,
                        symbol_count=len(symbols))
        
        start_time = time.time()
        
        try:
            # Call the actual model method
            analysis = self.model.analyze_market(symbols, timeframe)
            
            duration = time.time() - start_time
            
            # Record metrics
            self.logger.timing("market_analysis_time_ms", duration * 1000)
            self.logger.counter("market_analyses")
            self.logger.gauge("analyzed_symbols", len(symbols))
            
            # Log success with some analysis metrics
            self.logger.info("Market analysis completed", 
                           timeframe=timeframe,
                           symbol_count=len(symbols),
                           execution_time_ms=duration * 1000)
            
            return analysis
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            self.logger.counter("analysis_errors")
            
            # Log error
            self.logger.error("Market analysis failed", 
                             symbols=symbols, 
                             timeframe=timeframe,
                             error=str(e),
                             execution_time_ms=duration * 1000)
            raise
    
    def shutdown(self):
        """Shutdown the model and stop metrics collection"""
        self.logger.info("Shutting down trade model")
        self.logger.stop_system_metrics()
        
        # Call actual model shutdown if it exists
        if hasattr(self.model, 'shutdown'):
            self.model.shutdown()


class MonitoredRiskAssessment:
    """
    Example wrapper for RiskAssessmentModel that adds monitoring
    """
    
    def __init__(self, config_path=None):
        self.logger = NetdataLogger(component_name="risk-assessment")
        self.logger.info("Initializing risk assessment model")
        
        try:
            self.model = RiskAssessmentModel(config_path)
            self.logger.info("Risk assessment model initialized", config_path=config_path)
        except Exception as e:
            self.logger.error("Failed to initialize risk model", error=str(e))
            raise
    
    def assess_risk(self, portfolio, market_conditions):
        """Assess risk with monitoring"""
        self.logger.info("Assessing portfolio risk", 
                        portfolio_size=len(portfolio),
                        market_condition=market_conditions.get('condition', 'unknown'))
        
        start_time = time.time()
        
        try:
            # Call the actual risk assessment
            risk_profile = self.model.assess_risk(portfolio, market_conditions)
            
            duration = time.time() - start_time
            
            # Record metrics
            self.logger.timing("risk_assessment_time_ms", duration * 1000)
            self.logger.gauge("risk_score", risk_profile.get('risk_score', 0))
            self.logger.counter("risk_assessments")
            
            # Log the risk assessment results
            self.logger.info("Risk assessment completed", 
                           risk_score=risk_profile.get('risk_score', 0),
                           risk_level=risk_profile.get('risk_level', 'unknown'),
                           execution_time_ms=duration * 1000)
            
            return risk_profile
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.logger.error("Risk assessment failed", 
                             portfolio_size=len(portfolio),
                             error=str(e),
                             execution_time_ms=duration * 1000)
            raise


# Example of how to use these monitored components
def run_example():
    print("Starting integration example...")
    
    # Create monitored components
    trade_model = MonitoredTradeModel()
    risk_model = MonitoredRiskAssessment()
    
    # Example portfolio
    portfolio = [
        {"symbol": "AAPL", "quantity": 100, "price": 150.0},
        {"symbol": "MSFT", "quantity": 50, "price": 250.0},
        {"symbol": "GOOGL", "quantity": 20, "price": 2800.0}
    ]
    
    # Example market conditions
    market_conditions = {
        "condition": "volatile",
        "vix": 25.5,
        "trend": "bearish"
    }
    
    # Run risk assessment
    try:
        risk_profile = risk_model.assess_risk(portfolio, market_conditions)
        print(f"Risk assessment completed with score: {risk_profile.get('risk_score', 0)}")
    except Exception as e:
        print(f"Risk assessment failed: {e}")
    
    # Execute some trades
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    for symbol in symbols:
        try:
            # Random trade parameters
            quantity = random.randint(1, 100)
            price = random.uniform(100, 3000)
            side = random.choice(["buy", "sell"])
            
            # Execute trade
            result = trade_model.execute_trade(symbol, quantity, price, side)
            print(f"Trade executed: {symbol} {side} {quantity} @ ${price:.2f}")
            
        except Exception as e:
            print(f"Trade failed: {e}")
    
    # Analyze market
    try:
        analysis = trade_model.analyze_market(symbols, "1d")
        print(f"Market analysis completed for {len(symbols)} symbols")
    except Exception as e:
        print(f"Market analysis failed: {e}")
    
    # Shutdown
    trade_model.shutdown()
    print("Integration example completed")


if __name__ == "__main__":
    run_example()