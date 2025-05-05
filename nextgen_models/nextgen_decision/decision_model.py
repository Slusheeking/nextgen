"""
Decision Model for NextGen Trading System

This module implements the decision-making component of the NextGen Models system.
It aggregates data from all analysis models, applies risk management rules,
and makes final trading decisions.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid # Import uuid for generating unique IDs

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
)

# MCP tools (Consolidated)
# Import redis directly with graceful fallback
import importlib
try:
    import redis
except ImportError:
    redis = None

from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP # Import RedisMCP

# Import ContextModel for RAG
from nextgen_models.nextgen_context_model.context_model import ContextModel


class PortfolioAnalyzer:
    """Analyzes current portfolio state."""

    def __init__(self, decision_model: 'DecisionModel'): # Use forward declaration
        self.decision_model = decision_model
        self.logger = decision_model.logger
        self.redis_mcp = decision_model.redis_mcp # Access RedisMCP from DecisionModel
        self.redis_keys = decision_model.redis_keys # Access Redis keys from DecisionModel


    def get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio data using Redis."""
        self.logger.info("Getting portfolio data from Redis...")
        try:
            # Retrieve portfolio data from Redis (assuming TradeModel stores this)
            self.decision_model.mcp_tool_call_count += 1
            portfolio_data_result = self.redis_mcp.call_tool("get_json", {"key": self.redis_keys["portfolio_data"]})

            if portfolio_data_result and not portfolio_data_result.get("error"):
                portfolio_data = portfolio_data_result.get("value", {})
                self.logger.info(f"Retrieved portfolio data: {portfolio_data.get('equity')}")
                return portfolio_data
            elif portfolio_data_result and portfolio_data_result.get("error"):
                 self.decision_model.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to get portfolio data from Redis: {portfolio_data_result.get('error')}")
                 return {"error": portfolio_data_result.get('error', 'Failed to get portfolio data from Redis')}
            else:
                 self.logger.warning("No portfolio data found in Redis.")
                 return {"message": "No portfolio data found in Redis."}

        except Exception as e:
            self.logger.error(f"Error getting portfolio data from Redis: {e}")
            self.decision_model.execution_errors += 1
            self.decision_model.logger.counter("decision_model.execution_errors")
            return {"error": str(e)}


    def get_correlation_data(self) -> Dict[str, Any]:
        """Get asset correlation data from Redis or calculate it using RiskAnalysisMCP."""
        self.logger.info("Retrieving asset correlation data...")
        try:
            # First, try to get correlation data from Redis (cached)
            correlation_key = "portfolio:correlation:data"
            correlation_ttl = 3600  # Cache valid for 1 hour
            
            self.decision_model.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool("get_json", {"key": correlation_key})
            
            if result and not result.get("error") and result.get("value"):
                self.logger.info("Retrieved correlation data from Redis cache")
                return result.get("value")
            
            # If not found in Redis, calculate using RiskAnalysisMCP
            self.logger.info("Calculating correlation matrix via RiskAnalysisMCP...")
            
            # Get active positions for assets in portfolio
            portfolio_data = self.get_portfolio_data()
            positions = portfolio_data.get("positions", [])
            
            if not positions:
                self.logger.warning("No positions found to calculate correlations")
                return {"message": "No positions found", "correlation_matrix": {}}
            
            # Extract symbols from positions
            symbols = [position.get("symbol") for position in positions if position.get("symbol")]
            
            if not symbols:
                self.logger.warning("No symbols found in positions")
                return {"message": "No symbols found in positions", "correlation_matrix": {}}
            
            # Request correlation calculation from risk_analysis_mcp
            self.decision_model.mcp_tool_call_count += 1
            correlation_result = self.decision_model.risk_analysis_mcp.call_tool(
                "calculate_correlation_matrix", 
                {"symbols": symbols, "lookback_days": 90}  # 90-day lookback for correlation
            )
            
            if correlation_result and correlation_result.get("error"):
                self.decision_model.mcp_tool_error_count += 1
                self.logger.error(f"Error calculating correlation matrix: {correlation_result.get('error')}")
                return {"error": correlation_result.get("error")}
            
            # Store result in Redis for future use
            if correlation_result and correlation_result.get("correlation_matrix"):
                self.decision_model.mcp_tool_call_count += 1
                self.redis_mcp.call_tool(
                    "set_json", 
                    {
                        "key": correlation_key,
                        "value": correlation_result,
                        "expiry": correlation_ttl
                    }
                )
                self.logger.info(f"Stored correlation matrix in Redis with {len(symbols)} assets")
                
            return correlation_result if correlation_result else {"message": "Failed to calculate correlation matrix", "correlation_matrix": {}}
            
        except Exception as e:
            self.logger.error(f"Error retrieving correlation data: {e}")
            self.decision_model.execution_errors += 1
            self.decision_model.logger.counter("decision_model.execution_errors")
            return {"error": str(e), "correlation_matrix": {}}


class RiskManager:
    """Applies risk management rules."""

    def __init__(self, decision_model: 'DecisionModel'): # Use forward declaration
        self.decision_model = decision_model
        self.logger = decision_model.logger
        self.redis_mcp = decision_model.redis_mcp # Access RedisMCP from DecisionModel
        self.redis_keys = decision_model.redis_keys # Access Redis keys from DecisionModel
        self.max_position_pct = decision_model.max_position_size_pct
        self.risk_per_trade_pct = decision_model.risk_per_trade_pct
        # Default risk parameters
        self.default_confidence_level = 0.95  # 95% confidence level for VaR
        self.default_time_horizon = 1  # 1-day time horizon

    def _calculate_risk_contribution(
        self, 
        symbol: str, 
        position_size: float, 
        action: str,
        portfolio_data: Dict[str, Any],
        updated_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Calculate the marginal VaR and Expected Shortfall contribution of a new position.
        
        Args:
            symbol: The symbol of the asset being traded
            position_size: The size of the position
            action: The trade action (buy, sell)
            portfolio_data: The portfolio data
            updated_weights: Optional pre-calculated portfolio weights after the trade
            
        Returns:
            Tuple of (VaR contribution, ES contribution)
        """
        try:
            self.logger.info(f"Calculating risk contribution for {symbol} {action} {position_size}")
            
            # If no updated weights provided, calculate them
            if updated_weights is None:
                # Get current positions
                positions = portfolio_data.get("positions", [])
                
                # Find if position already exists
                existing_position = next((pos for pos in positions if pos.get("symbol") == symbol), None)
                latest_price = self.decision_model.get_latest_price(symbol)
                
                # Update positions based on the trade
                updated_positions = positions.copy()
                if action == "buy":
                    if existing_position:
                        # Update existing position
                        new_quantity = float(existing_position.get("quantity", 0)) + position_size
                        existing_position["quantity"] = new_quantity
                        existing_position["market_value"] = new_quantity * latest_price
                    else:
                        # Add new position
                        updated_positions.append({
                            "symbol": symbol,
                            "quantity": position_size,
                            "market_value": position_size * latest_price
                        })
                elif action == "sell" and existing_position:
                    # Reduce or remove position
                    new_quantity = float(existing_position.get("quantity", 0)) - position_size
                    if new_quantity <= 0:
                        # Remove position
                        updated_positions = [pos for pos in positions if pos.get("symbol") != symbol]
                    else:
                        # Update position with reduced quantity
                        existing_position["quantity"] = new_quantity
                        existing_position["market_value"] = new_quantity * latest_price
                
                # Calculate total position value after update
                total_position_value = sum(float(pos.get("market_value", 0)) for pos in updated_positions)
                
                # Create updated portfolio weights
                updated_weights = {}
                if total_position_value > 0:
                    for pos in updated_positions:
                        sym = pos.get("symbol")
                        market_value = float(pos.get("market_value", 0))
                        updated_weights[sym] = market_value / total_position_value
            
            # Get the portfolio VaR with the updated weights
            var_result = self.decision_model.risk_analysis_mcp.call_tool(
                "calculate_portfolio_risk", 
                {
                    "weights": updated_weights,
                    "risk_measure": "var_contribution",  # Request specifically VaR contribution
                    "target_symbol": symbol,              # Specify which symbol's contribution we want
                    "confidence_level": self.default_confidence_level,
                    "time_horizon": self.default_time_horizon
                }
            )
            
            # Get Expected Shortfall contribution
            es_result = self.decision_model.risk_analysis_mcp.call_tool(
                "calculate_portfolio_risk", 
                {
                    "weights": updated_weights,
                    "risk_measure": "es_contribution",    # Request ES contribution 
                    "target_symbol": symbol,              # Specify which symbol's contribution we want
                    "confidence_level": self.default_confidence_level,
                    "time_horizon": self.default_time_horizon
                }
            )
            
            # Extract the risk contributions
            var_contribution = 0.0
            es_contribution = 0.0
            
            if var_result and not var_result.get("error"):
                # Get the symbol's contribution to total VaR
                contributions = var_result.get("contributions", {})
                var_contribution = contributions.get(symbol, 0.0)
                
                # Convert from percentage to dollar amount
                equity = float(portfolio_data.get("equity", 0))
                var_contribution = var_contribution * equity
                
                self.logger.info(f"VaR contribution for {symbol}: ${var_contribution:.2f}")
            else:
                self.logger.warning(f"Error calculating VaR contribution: {var_result.get('error') if var_result else 'No result'}")
                # Fallback to simplified calculation if risk_analysis_mcp fails
                latest_price = self.decision_model.get_latest_price(symbol)
                var_contribution = position_size * latest_price * 0.05  # Approximate 5% potential loss
            
            if es_result and not es_result.get("error"):
                # Get the symbol's contribution to total Expected Shortfall
                contributions = es_result.get("contributions", {})
                es_contribution = contributions.get(symbol, 0.0)
                
                # Convert from percentage to dollar amount
                equity = float(portfolio_data.get("equity", 0))
                es_contribution = es_contribution * equity
                
                self.logger.info(f"ES contribution for {symbol}: ${es_contribution:.2f}")
            else:
                self.logger.warning(f"Error calculating ES contribution: {es_result.get('error') if es_result else 'No result'}")
                # Fallback to simplified calculation if risk_analysis_mcp fails
                latest_price = self.decision_model.get_latest_price(symbol)
                es_contribution = position_size * latest_price * 0.07  # Approximate 7% potential loss (higher than VaR)
            
            return var_contribution, es_contribution
            
        except Exception as e:
            self.logger.error(f"Error calculating risk contribution for {symbol}: {e}")
            # Fallback to simplified calculation
            latest_price = self.decision_model.get_latest_price(symbol)
            var_contribution = position_size * latest_price * 0.05  # Approximate 5% potential loss
            es_contribution = position_size * latest_price * 0.07   # Approximate 7% potential loss
            return var_contribution, es_contribution
    
    def check_risk_limits(
        self, symbol: str, position_size: float, action: str
    ) -> Dict[str, Any]:
        """Check if a proposed trade meets risk limits using Redis for portfolio data and limits."""
        self.logger.info(f"Checking risk limits for {symbol} {action} {position_size}")
        try:
            # Get current portfolio data from Redis
            portfolio_data = self.decision_model.get_portfolio_data() # Use DecisionModel method which uses Redis
            if "error" in portfolio_data:
                 self.logger.warning(f"Could not get portfolio data for risk check: {portfolio_data['error']}")
                 # Cannot perform full risk check without portfolio data
                 return {"approved": True, "reason": f"Cannot perform full risk check: {portfolio_data['error']}"}


            # Get risk limits from Redis using actual portfolio ID
            self.decision_model.mcp_tool_call_count += 1
            
            # Get portfolio ID from portfolio data
            portfolio_data = self.decision_model.get_portfolio_data()
            portfolio_id = portfolio_data.get("portfolio_id", "default")
            
            # Use properly formatted risk limits key with actual portfolio ID
            risk_limits_key = f"risk:limits:{portfolio_id}"
            self.logger.info(f"Retrieving risk limits for portfolio ID: {portfolio_id}")
            risk_limits_result = self.redis_mcp.call_tool("get_json", {"key": risk_limits_key})

            risk_limits = risk_limits_result.get("value", {}) if risk_limits_result and not risk_limits_result.get("error") else {}
            if risk_limits_result and risk_limits_result.get("error"):
                 self.decision_model.mcp_tool_error_count += 1
                 self.logger.warning(f"Could not get risk limits from Redis: {risk_limits_result.get('error')}")


            # Perform risk checks based on portfolio data and limits
            equity = float(portfolio_data.get("equity", 0))
            if equity <= 0:
                 return {"approved": False, "reason": "Invalid portfolio equity for risk check."}

            # Example: Check max position size percentage
            if action == "buy":
                current_position_value = float(portfolio_data.get("positions_value", 0)) # Assuming positions_value is in portfolio_data
                proposed_position_value = position_size * self.decision_model.get_latest_price(symbol) # Need latest price
                if proposed_position_value > equity * (self.max_position_pct / 100):
                    return {"approved": False, "reason": f"Proposed position size exceeds max position percentage ({self.max_position_pct}%)"}

            # Enhanced risk per trade check using VaR/ES contribution calculation
            # Calculate the marginal contribution of the new position to portfolio risk
            var_contribution, es_contribution = self._calculate_risk_contribution(
                symbol, position_size, action, portfolio_data, updated_weights
            )
            
            # Check if VaR contribution exceeds risk per trade limit
            if var_contribution > equity * (self.risk_per_trade_pct / 100):
                return {"approved": False, "reason": f"Position VaR contribution ({var_contribution:.2f}) exceeds risk per trade percentage ({self.risk_per_trade_pct}% of equity)"}
                
            # Also check Expected Shortfall (ES) contribution as it captures tail risk better
            if es_contribution > equity * (self.risk_per_trade_pct * 1.5 / 100):  # Allow slightly higher ES
                return {"approved": False, "reason": f"Position Expected Shortfall contribution ({es_contribution:.2f}) exceeds risk per trade limit ({self.risk_per_trade_pct * 1.5}% of equity)"}


            # Perform comprehensive risk checks against retrieved risk limits
            if risk_limits:
                self.logger.info(f"Performing comprehensive risk checks with position: {symbol} {action} {position_size}")
                
                # 1. Check sector exposure if adding a new position
                if action == "buy":
                    # Get sector for the symbol
                    self.decision_model.mcp_tool_call_count += 1
                    sector_info_result = self.decision_model.risk_analysis_mcp.call_tool(
                        "get_symbol_metadata", 
                        {"symbol": symbol}
                    )
                    
                    if sector_info_result and not sector_info_result.get("error"):
                        sector = sector_info_result.get("sector", "Unknown")
                        
                        # Calculate current sector exposure
                        positions = portfolio_data.get("positions", [])
                        sector_exposure = 0
                        equity = float(portfolio_data.get("equity", 0))
                        
                        # Sum up existing positions in the same sector
                        for pos in positions:
                            if pos.get("sector") == sector:
                                sector_exposure += float(pos.get("market_value", 0))
                        
                        # Add the new position value
                        new_position_value = position_size * self.decision_model.get_latest_price(symbol)
                        new_sector_exposure = (sector_exposure + new_position_value) / equity if equity > 0 else 0
                        
                        # Check against sector exposure limit
                        max_sector_exposure = risk_limits.get("max_sector_exposure", 0.25)  # Default 25%
                        if new_sector_exposure > max_sector_exposure:
                            return {
                                "approved": False, 
                                "reason": f"Position would cause {sector} sector exposure of {new_sector_exposure:.1%}, exceeding limit of {max_sector_exposure:.1%}"
                            }
                        
                        self.logger.info(f"Sector exposure check passed: {sector} at {new_sector_exposure:.1%} (limit: {max_sector_exposure:.1%})")
                
                # 2. Check portfolio VaR after adding/modifying position
                try:
                    # Calculate updated portfolio weights with the proposed trade
                    updated_positions = portfolio_data.get("positions", []).copy()
                    
                    # Find if position already exists
                    existing_position = next((pos for pos in updated_positions if pos.get("symbol") == symbol), None)
                    latest_price = self.decision_model.get_latest_price(symbol)
                    
                    if action == "buy":
                        if existing_position:
                            # Update existing position
                            new_quantity = float(existing_position.get("quantity", 0)) + position_size
                            existing_position["quantity"] = new_quantity
                            existing_position["market_value"] = new_quantity * latest_price
                        else:
                            # Add new position
                            updated_positions.append({
                                "symbol": symbol,
                                "quantity": position_size,
                                "market_value": position_size * latest_price
                            })
                    elif action == "sell" and existing_position:
                        # Reduce or remove position
                        new_quantity = float(existing_position.get("quantity", 0)) - position_size
                        if new_quantity <= 0:
                            # Remove position
                            updated_positions = [pos for pos in updated_positions if pos.get("symbol") != symbol]
                        else:
                            # Update position with reduced quantity
                            existing_position["quantity"] = new_quantity
                            existing_position["market_value"] = new_quantity * latest_price
                    
                    # Calculate total position value after update
                    total_position_value = sum(float(pos.get("market_value", 0)) for pos in updated_positions)
                    
                    # Create updated portfolio weights
                    updated_weights = {}
                    if total_position_value > 0:
                        for pos in updated_positions:
                            pos_symbol = pos.get("symbol")
                            market_value = float(pos.get("market_value", 0))
                            updated_weights[pos_symbol] = market_value / total_position_value
                    
                    # Calculate VaR with updated portfolio and get VaR/ES contributions for the symbol
                    self.decision_model.mcp_tool_call_count += 1
                    
                    # First, calculate the standard portfolio VaR
                    var_result = self.decision_model.risk_analysis_mcp.call_tool(
                        "calculate_portfolio_risk", 
                        {
                            "weights": updated_weights,
                            "risk_measure": "var",
                            "confidence_level": self.default_confidence_level,
                            "time_horizon": self.default_time_horizon
                        }
                    )
                    
                    # Then, calculate the VaR and ES contributions of the symbol
                    var_contribution, es_contribution = self._calculate_risk_contribution(
                        symbol, position_size, action, portfolio_data, updated_weights
                    )
                    
                    # Log the risk contributions
                    self.logger.info(f"VaR contribution for {symbol}: ${var_contribution:.2f}")
                    self.logger.info(f"ES contribution for {symbol}: ${es_contribution:.2f}")
                    
                    if var_result and not var_result.get("error"):
                        portfolio_var = var_result.get("var", 0)
                        max_var_limit = risk_limits.get("max_var", 0.03)  # Default 3% VaR limit
                        
                        if portfolio_var > max_var_limit:
                            return {
                                "approved": False, 
                                "reason": f"Position would increase portfolio VaR to {portfolio_var:.1%}, exceeding limit of {max_var_limit:.1%}"
                            }
                        
                        self.logger.info(f"VaR check passed: {portfolio_var:.2%} (limit: {max_var_limit:.2%})")
                    
                    # 3. Check maximum drawdown risk
                    self.decision_model.mcp_tool_call_count += 1
                    drawdown_result = self.decision_model.risk_analysis_mcp.call_tool(
                        "calculate_portfolio_risk", 
                        {
                            "weights": updated_weights,
                            "risk_measure": "expected_shortfall",  # Conditional VaR / Expected Shortfall
                            "confidence_level": 0.95,
                            "time_horizon": 1
                        }
                    )
                    
                    if drawdown_result and not drawdown_result.get("error"):
                        expected_shortfall = drawdown_result.get("expected_shortfall", 0)
                        max_es_limit = risk_limits.get("max_expected_shortfall", 0.05)  # Default 5% ES limit
                        
                        if expected_shortfall > max_es_limit:
                            return {
                                "approved": False, 
                                "reason": f"Position would increase Expected Shortfall to {expected_shortfall:.1%}, exceeding limit of {max_es_limit:.1%}"
                            }
                        
                        self.logger.info(f"Expected Shortfall check passed: {expected_shortfall:.2%} (limit: {max_es_limit:.2%})")
                    
                except Exception as e:
                    self.logger.error(f"Error in portfolio risk calculation: {e}")
                    # Don't block the trade on calculation error, but log it
                    self.decision_model.execution_errors += 1


            return {"approved": True, "reason": "Risk checks passed"}

        except Exception as e:
            self.logger.error(f"Error checking risk limits for {symbol}: {e}")
            self.decision_model.execution_errors += 1
            self.decision_model.logger.counter("decision_model.execution_errors")
            return {"approved": False, "reason": f"Error during risk check: {str(e)}"}


class MarketStateAnalyzer:
    """Evaluates overall market conditions."""

    def __init__(self, decision_model: 'DecisionModel'): # Use forward declaration
        self.decision_model = decision_model
        self.logger = decision_model.logger

    def evaluate_market_conditions(self) -> Dict[str, Any]:
        """Evaluate market conditions using FinancialDataMCP."""
        start_time = time.time()
        try:
            # Get market status (using FinancialDataMCP tool)
            self.decision_model.mcp_tool_call_count += 1
            market_status_result = self.decision_model.financial_data_mcp.call_tool("get_market_status", {})
            is_market_open = False
            if market_status_result and not market_status_result.get("error"):
                 is_market_open = market_status_result.get("market", "closed") == "open"
            elif market_status_result and market_status_result.get("error"):
                 self.decision_model.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting market status: {market_status_result['error']}")


            # Get VIX (using FinancialDataMCP tool)
            self.decision_model.mcp_tool_call_count += 1
            vix_result = self.decision_model.financial_data_mcp.call_tool("get_latest_trade", {"symbol": "VIX"})
            vix_value = 15.0 # Default
            if vix_result and not vix_result.get("error"):
                 vix_value = vix_result.get("price", 15.0)
            elif vix_result and vix_result.get("error"):
                 self.decision_model.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting VIX data: {vix_result['error']}")


            # Get SPY price (using FinancialDataMCP tool)
            self.decision_model.mcp_tool_call_count += 1
            spy_result = self.decision_model.financial_data_mcp.call_tool("get_latest_trade", {"symbol": "SPY"})
            spy_price = None
            if spy_result and not spy_result.get("error"):
                 spy_price = spy_result.get("price")
            elif spy_result and spy_result.get("error"):
                 self.decision_model.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting SPY data: {spy_result['error']}")


            # Determine volatility state
            volatility = "normal"
            if vix_value > 25:
                volatility = "high"
            elif vix_value < 15:
                volatility = "low"

            # Determine market state using RiskAnalysisMCP drift detection
            market_state = "neutral"
            try:
                # Call drift detection on RiskAnalysisMCP
                self.decision_model.mcp_tool_call_count += 1
                drift_result = self.decision_model.risk_analysis_mcp.call_tool(
                    "detect_drift", 
                    {
                        "symbol": "SPY", 
                        "lookback_days": 30,
                        "window_size": 20,
                        "z_threshold": 2.0
                    }
                )
                
                if drift_result and not drift_result.get("error"):
                    market_state = drift_result.get("trend", "neutral")
                    self.logger.info(f"Drift detection result: {market_state}")
                    
                    # Get additional market indicators
                    self.decision_model.mcp_tool_call_count += 1
                    market_indicators = self.decision_model.financial_data_mcp.call_tool(
                        "get_market_indicators", 
                        {
                            "symbols": ["SPY", "QQQ", "DIA", "IWM", "VIX"],
                            "indicators": ["sma_50", "sma_200", "rsi", "macd"]
                        }
                    )
                    
                    if market_indicators and not market_indicators.get("error"):
                        # Analyze market indicators to refine market state
                        spy_indicators = market_indicators.get("SPY", {})
                        
                        # Check if SPY is above/below key moving averages
                        spy_price = spy_result.get("price") if spy_result else None
                        sma_50 = spy_indicators.get("sma_50")
                        sma_200 = spy_indicators.get("sma_200")
                        
                        if spy_price and sma_50 and sma_200:
                            if spy_price > sma_50 and spy_price > sma_200:
                                if market_state == "neutral":
                                    market_state = "bullish"
                            elif spy_price < sma_50 and spy_price < sma_200:
                                if market_state == "neutral":
                                    market_state = "bearish"
                        
                        # Check RSI for overbought/oversold conditions
                        rsi = spy_indicators.get("rsi")
                        if rsi:
                            if rsi > 70 and market_state == "bullish":
                                market_state = "overbought"
                            elif rsi < 30 and market_state == "bearish":
                                market_state = "oversold"
                else:
                    self.decision_model.mcp_tool_error_count += 1
                    self.logger.warning(f"Failed to get drift detection result: {drift_result.get('error') if drift_result else 'No result'}")
            except Exception as e:
                self.logger.error(f"Error in market state detection: {e}")
                # Continue with default neutral state

            result = {
                "is_market_open": is_market_open,
                "volatility_index": vix_value,
                "volatility_state": volatility,
                "market_state": market_state,
                "spy_price": spy_price,
                "timestamp": datetime.now().isoformat(),
            }
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.market_state_analyzer.evaluate_market_conditions_duration_ms", duration, tags={"status": "success"})
            return result
        except Exception as e:
            self.logger.error(f"Error evaluating market conditions: {e}")
            self.decision_model.execution_errors += 1
            self.decision_model.logger.counter("decision_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.market_state_analyzer.evaluate_market_conditions_duration_ms", duration, tags={"status": "failed"})
            return {"error": str(e)}


class DecisionModel:
    """
    Decision Model for the NextGen Trading System.

    Aggregates data from all analysis models, considers portfolio constraints
    and market conditions, applies risk management rules, and makes final
    trading decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Decision Model.

        Args:
            config: Optional configuration dictionary
        """
        init_start_time = time.time()
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-decision-model")

        # Initialize StockChartGenerator
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")

        # Initialize counters for decision model metrics
        self.decisions_made_count = 0
        self.buy_decisions_count = 0
        self.sell_decisions_count = 0
        self.hold_decisions_count = 0
        self.risk_rejections_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0 # Errors during decision making process
        self.total_decision_cycles = 0 # Total times process_analysis_results is called


        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_decision", "decision_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}")
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}")
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}")
                self.config = {}
        else:
            self.config = config

        # Initialize Consolidated MCP clients
        # RiskAnalysisMCP handles decision analytics, portfolio optimization, and drift detection
        self.risk_analysis_mcp = RiskAnalysisMCP(
             self.config.get("risk_analysis_config")
        )
        # FinancialDataMCP handles data retrieval (Polygon REST)
        self.financial_data_mcp = FinancialDataMCP(
            self.config.get("financial_data_config")
        )
        # Initialize Redis MCP client for inter-model communication and decision storage
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))


        # Initialize ContextModel for RAG functionality
        # Pass relevant parts of the config if needed (e.g., vector_store_config, document_analysis_config)
        context_model_config = {
            "vector_store_config": self.config.get("vector_store_config"),
            "document_analysis_config": self.config.get("document_analysis_config"),
            "llm_config": self.config.get("llm_config") # ContextModel might need its own LLM config
        }
        self.context_model = ContextModel(config=context_model_config)
        self.logger.info("Initialized ContextModel for RAG")


        # Initialize helper component classes, passing self for MCP/config access
        self.portfolio_analyzer = PortfolioAnalyzer(self)
        self.risk_manager = RiskManager(self)
        self.market_state_analyzer = MarketStateAnalyzer(self)


        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        # Redis keys for data access and storage - Production stream and key names
        self.redis_keys = {
            # Keys and streams for accessing model reports (using Redis for inter-model communication)
            "sentiment_reports_stream": "nextgen:model:sentiment:analysis:stream",
            "sentiment_data_key": "nextgen:model:sentiment:data:",  # Append symbol for specific symbol data
            "sentiment_market_key": "nextgen:model:sentiment:market",  # Overall market sentiment
            
            "fundamental_reports_stream": "nextgen:model:fundamental:insights:stream",
            "fundamental_data_key": "nextgen:model:fundamental:data:",  # Append symbol for specific symbol data
            "fundamental_market_key": "nextgen:model:fundamental:market",  # Overall market fundamental data
            
            "market_analysis_stream": "nextgen:model:market:analysis:stream",
            "market_data_key": "nextgen:model:market:data",  # Market analysis data
            "market_trends_key": "nextgen:model:market:trends",  # Market trends
            
            "technical_reports_stream": "nextgen:model:technical:analysis:stream",
            "technical_data_key": "nextgen:model:technical:data:",  # Append symbol for specific symbol data
            
            "selection_requests_stream": "nextgen:model:selection:requests:stream",  # Stream for selection requests
            "selection_reports_stream": "nextgen:model:selection:candidates:stream",  # Stream for selection results
            "selection_data_key": "nextgen:model:selection:candidates",  # Latest selection candidates
            
            "risk_package_stream": "nextgen:model:risk:assessments:stream",
            "risk_assessment_key": "nextgen:model:risk:assessment:",  # Append symbol for specific symbol data
            
            # Keys for storing decisions
            "decisions_stream": "nextgen:decision:stream",  # Stream of all decisions
            "decisions_list": "nextgen:decision:list",  # List of recent decisions
            "decision_history": "nextgen:decision:history",  # Sorted set for history
            "latest_decision": "nextgen:decision:latest:",  # Append symbol for latest decision per symbol
            
            # Keys for accessing TradeModel data
            "account_info": "nextgen:trade:account_info",  # Latest account info from TradeModel
            "portfolio_data": "nextgen:trade:portfolio_data",  # Latest portfolio summary
            "positions": "nextgen:trade:positions",  # Latest list of positions
            "capital_available": "nextgen:trade:capital_available",  # Available capital
            "order_history": "nextgen:trade:order_history",  # History of orders
            "execution_history": "nextgen:trade:execution_history",  # History of executions
            
            # Keys for other data
            "correlation_data": "nextgen:portfolio:correlation:data",  # Asset correlation data
            "risk_limits": "nextgen:risk:limits:",  # Append portfolio ID for specific limits
        }

        # Ensure Redis streams exist (optional, but good practice)
        try:
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["risk_package_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['risk_package_stream']}' exists.")
        except Exception as e:
            self.logger.warning(f"Could not ensure Redis stream exists: {e}")


        # Decision thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.max_position_size_pct = self.config.get("max_position_size_pct", 5.0)
        self.risk_per_trade_pct = self.config.get("risk_per_trade_pct", 1.0)

        self.logger.info("Decision Model initialized")
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("decision_model.initialization_time_ms", init_duration)


    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen.

        Returns:
            LLM configuration dictionary
        """
        llm_config = self.config.get("llm_config", {})

        # Default configuration if not provided
        if not llm_config:
            llm_config = {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    },
                    {
                        "model": "meta-llama/llama-3-70b-instruct",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    },
                ],
            }

        return {
            "config_list": llm_config.get("config_list", []),
            "temperature": llm_config.get("temperature", 0.1),
            "timeout": llm_config.get("timeout", 600),
            "seed": 42,  # Adding seed for reproducibility
        }

    def _setup_agents(self) -> Dict[str, Agent]:
        """
        Initialize AutoGen agents for decision making.

        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the decision assistant agent
        agents["decision_assistant"] = AssistantAgent(
            name="DecisionAssistantAgent",
            system_message="""You are a trading decision specialist. Your role is to:
            1. Analyze data from multiple sources (selection, sentiment, forecasting, fundamental analysis, RAG context, etc.)
            2. Consider portfolio constraints and risk management rules
            3. Make final trading decisions with confidence levels
            4. Provide clear reasoning for each decision

            You have tools for calculating confidence scores, optimizing position sizes,
            analyzing portfolio impact, retrieving RAG context, and evaluating market conditions.
            Always consider risk management first and provide detailed reasoning for your decisions.

            Pay special attention to fundamental analysis data, which provides insights on:
            - Financial health scores
            - Growth quality and sustainability
            - Valuation metrics
            - Sector comparisons""",
            llm_config=self.llm_config,
            description="A specialist in trading decision making",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="DecisionToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        agents["user_proxy"] = user_proxy

        return agents

    def _register_functions(self):
        """
        Register functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        decision_assistant = self.agents["decision_assistant"]

        # Define data access functions (Implemented using Redis)
        @register_function(
            name="get_selection_data",
            description="Get data from the Selection Model via Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_selection_data() -> Dict[str, Any]:
            start_time = time.time()
            result = self.get_selection_data() # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_selection_data"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_selection_data"})
            return result

        @register_function(
            name="get_finnlp_data",
            description="Get data from the FinNLP Model via Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_finnlp_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            start_time = time.time()
            result = self.get_finnlp_data(symbol) # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_finnlp_data"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_finnlp_data"})
            return result

        @register_function(
            name="get_forecaster_data",
            description="Get data from the Forecaster Model via Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_forecaster_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            start_time = time.time()
            result = self.get_forecaster_data(symbol) # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_forecaster_data"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_forecaster_data"})
            return result

        @register_function(
            name="get_rag_context", # Renamed for clarity
            description="Retrieve relevant contextual documents for a symbol using RAG.",
            caller=decision_assistant,
            executor=user_proxy,
        )
        async def get_rag_context( # Made async as retrieve_context is async
            symbol: str,
            collection_name: Optional[str] = None,
            top_k: int = 5 # Default to fewer results for context
            ) -> Dict[str, Any]:
            start_time = time.time()
            # Call the integrated ContextModel's retrieve_context method
            result = await self.context_model.retrieve_context(
                query=symbol, # Use symbol as the query for now
                collection_name=collection_name,
                top_k=top_k
            )
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_rag_context"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_rag_context"})
            # Return the retrieved context directly or format as needed
            return result.get("retrieved_context", []) if result and "error" not in result else {"error": result.get("error", "Failed to retrieve RAG context")}


        @register_function(
            name="get_fundamental_data",
            description="Get data from the Fundamental Analysis Model via Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_fundamental_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            start_time = time.time()
            result = self.get_fundamental_data(symbol) # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_fundamental_data"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_fundamental_data"})
            return result

        @register_function(
            name="get_portfolio_data",
            description="Get current portfolio data via Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_portfolio_data() -> Dict[str, Any]:
            start_time = time.time()
            result = self.portfolio_analyzer.get_portfolio_data() # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_portfolio_data"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_portfolio_data"})
            return result

        # Define decision-making functions (using RiskAnalysisMCP)
        @register_function(
            name="calculate_confidence_score",
            description="Calculate confidence score from multiple signals",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def calculate_confidence_score(
            signals: Dict[str, Any], weights: Optional[Dict[str, float]] = None
        ) -> Dict[str, Any]:
            start_time = time.time()
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool(
                "calculate_confidence_score",
                {"signals": signals, "weights": weights, "threshold": self.confidence_threshold}
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_confidence_score_duration_ms", duration, tags={"function": "calculate_confidence_score"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "calculate_confidence_score"})
            return result

        @register_function(
            name="calculate_optimal_position_size",
            description="Calculate optimal position size based on confidence and portfolio constraints",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def calculate_optimal_position_size(
            symbol: str, confidence: float
        ) -> Dict[str, Any]:
            start_time = time.time()
            try: # Added try block
                portfolio_data = self.portfolio_analyzer.get_portfolio_data() # Get current portfolio data
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "calculate_optimal_position_size",
                    {
                        "symbol": symbol, "confidence": confidence, "portfolio_data": portfolio_data,
                        "max_position_pct": self.max_position_size_pct, "risk_per_trade_pct": self.risk_per_trade_pct,
                    }
                )
                if result and result.get("error"):
                     self.mcp_tool_error_count += 1
                duration = (time.time() - start_time) * 1000
                self.logger.timing("decision_model.calculate_optimal_position_size_duration_ms", duration, tags={"status": "success"})
                return result
            except Exception as e: # Added except block
                self.logger.error(f"Error calculating optimal position size for {symbol}: {e}")
                self.execution_errors += 1
                self.logger.counter("decision_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("decision_model.calculate_optimal_position_size_duration_ms", duration, tags={"status": "failed"})
                return {"error": str(e)}


        @register_function(
            name="calculate_portfolio_impact",
            description="Calculate the impact of a new position on the portfolio",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def calculate_portfolio_impact(
            symbol: str, position_size: float
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                portfolio_data = self.portfolio_analyzer.get_portfolio_data()
                correlation_data = self.portfolio_analyzer.get_correlation_data()
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "calculate_portfolio_impact",
                    {
                        "symbol": symbol, "position_size": position_size,
                        "portfolio_data": portfolio_data, "correlation_data": correlation_data,
                    }
                )
                if result and result.get("error"): self.mcp_tool_error_count += 1
                duration = (time.time() - start_time) * 1000
                self.logger.timing("decision_model.calculate_portfolio_impact_duration_ms", duration, tags={"status": "success"})
                return result
            except Exception as e:
                self.logger.error(f"Error calculating portfolio impact for {symbol}: {e}")
                self.execution_errors += 1
                self.logger.counter("decision_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("decision_model.calculate_portfolio_impact_duration_ms", duration, tags={"status": "failed"})
                return {"error": str(e)}


        @register_function(
            name="store_decision",
            description="Store a trading decision in Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def store_decision(decision: Dict[str, Any]) -> bool:
            start_time = time.time()
            result = self.store_decision(decision) # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "store_decision"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "store_decision"})
            return result

        @register_function(
            name="get_recent_decisions",
            description="Get recent trading decisions from Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_recent_decisions(limit: int = 10) -> List[Dict[str, Any]]:
            start_time = time.time()
            result = self.get_recent_decisions(limit) # Calls the implemented method
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.function_call_duration_ms", duration, tags={"function": "get_recent_decisions"})
            self.logger.counter("decision_model.function_call_count", tags={"function": "get_recent_decisions"})
            return result

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        decision_assistant = self.agents["decision_assistant"]

        # Define MCP tool access functions for consolidated MCPs
        @register_function(
            name="use_risk_analysis_tool",
            description="Use a tool provided by the Risk Analysis MCP server (for decision analytics, optimization, drift, etc.)",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_risk_analysis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="use_financial_data_tool",
            description="Use a tool provided by the Financial Data MCP server (for market data)",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_financial_data_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.financial_data_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        # Old MCP tool access functions removed

    def process_analysis_results(
        self,
        selection_data: Dict[str, Any],
        finnlp_data: Dict[str, Any],
        forecaster_data: Dict[str, Any],
        # rag_data is no longer passed directly, fetched via get_rag_context tool
        fundamental_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process analysis results from all models to make decisions.
        Relies on AutoGen agent to call get_rag_context tool as needed.

        Args:
            selection_data: Data from the Selection Model
            finnlp_data: Data from the FinNLP Model
            forecaster_data: Data from the Forecaster Model
            fundamental_data: Data from the Fundamental Analysis Model (optional)

        Returns:
            List of trading decisions
        """
        self.logger.info("Processing analysis results...")
        self.total_decision_cycles += 1
        start_time = time.time()

        # Get candidates from selection data (placeholder)
        candidates = selection_data.get("candidates", [])
        if not candidates:
            self.logger.warning("No candidates found in selection data")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.process_analysis_results_duration_ms", duration, tags={"status": "no_candidates"})
            return []

        # Get market conditions (using FinancialDataMCP)
        market_conditions = self.market_state_analyzer.evaluate_market_conditions()

        # Get portfolio constraints (placeholder)
        portfolio_data = self.portfolio_analyzer.get_portfolio_data()

        # Process each candidate
        decisions = []
        for candidate in candidates:
            symbol = candidate.get("symbol") # Assuming 'symbol' key exists
            if not symbol:
                continue

            # Gather all analysis data for this symbol (placeholders for Redis data)
            # RAG data will be fetched by the agent using the get_rag_context tool
            symbol_analysis = {
                "selection": candidate,
                "finnlp": self.get_finnlp_data(symbol), # Placeholder call - now uses Redis
                "forecaster": self.get_forecaster_data(symbol), # Placeholder call - now uses Redis
                "fundamental": self.get_fundamental_data(symbol) if fundamental_data else {}, # Placeholder call - now uses Redis
                "market_conditions": market_conditions,
                "portfolio_data": portfolio_data, # Pass portfolio context
            }

            # Make decision for this symbol using AutoGen agent
            # The agent will use registered tools like get_rag_context, calculate_confidence_score etc.
            decision = self.make_trade_decision(symbol, symbol_analysis)

            # Apply risk management (using internal RiskManager component)
            if decision.get("action") != "hold":
                risk_check = self.risk_manager.check_risk_limits(
                    symbol, decision.get("position_size", 0), decision.get("action", "")
                )

                if not risk_check.get("approved", False):
                    self.logger.warning(
                        f"Trade for {symbol} rejected by risk management: {risk_check.get('reason')}"
                    )
                    self.risk_rejections_count += 1
                    self.logger.counter("decision_model.risk_rejections_count")
                    decision["action"] = "hold"
                    decision["reason"] = f"Risk management: {risk_check.get('reason')}"
                else:
                    # Ensure position size is valid after risk check
                    if decision.get("position_size", 0) <= 0 and decision.get("action") == "buy":
                        self.logger.warning(
                            f"Invalid position size (<=0) for {symbol} after risk check."
                        )
                        self.risk_rejections_count += 1 # Count as a risk rejection due to invalid size
                        self.logger.counter("decision_model.risk_rejections_count")
                        decision["action"] = "hold"
                        decision["reason"] = "Invalid position size after risk check"

            # Store the decision
            if decision:
                self.store_decision(decision) # Now uses Redis
                # Only add actionable decisions to the list to be executed
                if decision.get("action") != "hold":
                    decisions.append(decision)

        self.logger.info(f"Made {len(decisions)} actionable trading decisions")
        self.decisions_made_count += len(decisions)
        self.logger.counter("decision_model.actionable_decisions_count", len(decisions))
        self.logger.gauge("decision_model.total_decisions_made", self.decisions_made_count)
        self.logger.gauge("decision_model.risk_rejection_rate", (self.risk_rejections_count / self.total_decision_cycles) * 100 if self.total_decision_cycles > 0 else 0)


        duration = (time.time() - start_time) * 1000
        self.logger.timing("decision_model.process_analysis_results_duration_ms", duration, tags={"status": "success"})

        return decisions


    def _get_symbol_data(
        self, model_data: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Extract symbol-specific data from model data.
        (This helper might be less needed now that data is fetched via registered tools)
        """
        # This function might not be needed if data is fetched via registered tools
        self.logger.warning(f"Attempted to get symbol data for {symbol} via _get_symbol_data. Consider using registered tools instead.")
        return {}

    def make_trade_decision(
        self, symbol: str, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a trade decision for a specific symbol using AutoGen agent.

        Args:
            symbol: Stock symbol
            analysis_data: Combined analysis data from all models (should contain data for this symbol)

        Returns:
            Trading decision
        """
        self.logger.info(f"Making trade decision for {symbol} using AutoGen agent")
        start_time = time.time()

        decision_assistant = self.agents["decision_assistant"]
        user_proxy = self.agents["user_proxy"]

        if not decision_assistant or not user_proxy:
             self.logger.error("AutoGen agents not initialized for make_trade_decision")
             self.execution_errors += 1
             self.logger.counter("decision_model.execution_errors")
             duration = (time.time() - start_time) * 1000
             self.logger.timing("decision_model.make_trade_decision_duration_ms", duration, tags={"status": "failed", "reason": "agents_not_initialized"})
             return {
                "symbol": symbol,
                "action": "hold",
                "reasoning": "AutoGen agents not initialized",
                "timestamp": datetime.now().isoformat(),
            }


        # Prepare the prompt for the decision assistant
        # The agent should use get_rag_context(symbol=...) tool internally if needed
        prompt = f"""
        Analyze the following data for symbol {symbol} and make a trading decision (buy, sell, or hold).
        Provide confidence level (0-1) and reasoning. If buying, suggest an optimal position size.

        Analysis Data (Note: RAG context needs to be fetched using the get_rag_context tool if required):
        {json.dumps(analysis_data, indent=2)}

        Use the available tools to:
        1. Fetch relevant RAG context for {symbol} if needed using get_rag_context.
        2. Calculate a combined confidence score based on all available data (including RAG).
        3. Evaluate market conditions.
        4. Analyze fundamental data (financial health, growth quality, valuation).
        5. If confidence is high enough, determine the action (buy/sell).
        6. If buying, calculate the optimal position size considering risk and portfolio constraints.
        7. Check risk limits for the proposed trade.
        8. Format the final decision clearly.

        Example Decision Format:
        {{
            "symbol": "{symbol}",
            "action": "buy/sell/hold",
            "confidence": 0.85,
            "position_size": 100.0, // Only if action is 'buy'
            "reasoning": "Detailed explanation incorporating all relevant data sources...",
            "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffff"
        }}
        """

        # Initiate chat with the decision assistant
        try:
            llm_call_start_time = time.time()
            user_proxy.initiate_chat(decision_assistant, message=prompt)
            llm_call_duration = (time.time() - llm_call_start_time) * 1000
            self.logger.timing("decision_model.llm_call_duration_ms", llm_call_duration)
            self.llm_api_call_count += 1
            self.logger.counter("decision_model.llm_api_call_count")

            # Get the last message from the assistant
            last_message = user_proxy.last_message(decision_assistant)
            content = last_message.get("content", "")
            
            # Attempt to parse the decision from the content
            try:
                # Find the JSON block in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    decision_str = content[json_start:json_end]
                    decision = json.loads(decision_str)
                    # Add timestamp if missing
                    if "timestamp" not in decision:
                        decision["timestamp"] = datetime.now().isoformat()
                    # Add symbol if missing
                    if "symbol" not in decision:
                        decision["symbol"] = symbol
                    # Ensure position size is float
                    if "position_size" in decision:
                        decision["position_size"] = float(
                            decision.get("position_size", 0.0)
                        )
                    else:
                        # Default position size to 0 if not buying
                        if decision.get("action") != "buy":
                            decision["position_size"] = 0.0

                    self.logger.info(
                        f"Decision made for {symbol}: {decision.get('action')}"
                    )
                    self.logger.gauge("decision_model.confidence_score", decision.get("confidence", 0.0), tags={"symbol": symbol})
                    action = decision.get("action")
                    if action == "buy":
                        self.buy_decisions_count += 1
                        self.logger.counter("decision_model.buy_decisions_count")
                    elif action == "sell":
                        self.sell_decisions_count += 1
                        self.logger.counter("decision_model.sell_decisions_count")
                    else:
                        self.hold_decisions_count += 1
                        self.logger.counter("decision_model.hold_decisions_count")

                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.make_trade_decision_duration_ms", duration, tags={"status": "success", "action": action})

                    return decision
                else:
                    self.logger.warning(
                        f"Could not parse JSON decision from agent response for {symbol}. Content: {content}"
                    )
                    self.execution_errors += 1
                    self.logger.counter("decision_model.execution_errors")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.make_trade_decision_duration_ms", duration, tags={"status": "failed", "reason": "json_parse_error"})

                    return {
                        "symbol": symbol,
                        "action": "hold",
                        "reasoning": "Failed to parse agent decision",
                        "timestamp": datetime.now().isoformat(),
                    }

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Error decoding JSON decision for {symbol}: {e}. Content: {content}"
                )
                self.execution_errors += 1
                self.logger.counter("decision_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("decision_model.make_trade_decision_duration_ms", duration, tags={"status": "failed", "reason": "json_decode_error"})

                return {
                    "symbol": symbol,
                    "action": "hold",
                    "reasoning": f"JSON decode error: {e}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error during AutoGen chat for {symbol}: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.make_trade_decision_duration_ms", duration, tags={"status": "failed", "reason": "autogen_chat_error"})

            return {
                "symbol": symbol,
                "action": "hold",
                "reasoning": f"AutoGen chat error: {e}",
                "timestamp": datetime.now().isoformat(),
            }

    def calculate_confidence_score(
        self, signals: Dict[str, Any], weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate a confidence score from multiple signals using RiskAnalysisMCP.
        (Code unchanged)
        """
        start_time = time.time()
        try:
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool(
                "calculate_confidence_score",
                {"signals": signals, "weights": weights, "threshold": self.confidence_threshold}
            )
            if result and result.get("error"): self.mcp_tool_error_count += 1
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_confidence_score_duration_ms", duration, tags={"status": "success"})
            return result
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_confidence_score_duration_ms", duration, tags={"status": "failed"})
            return {"error": str(e)}


    def calculate_optimal_position_size(
        self, symbol: str, confidence: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on confidence and portfolio constraints using RiskAnalysisMCP.
        (Code unchanged)
        """
        start_time = time.time()
        try:
            portfolio_data = self.portfolio_analyzer.get_portfolio_data() # Get current portfolio data
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool(
                "calculate_optimal_position_size",
                {
                    "symbol": symbol, "confidence": confidence, "portfolio_data": portfolio_data,
                    "max_position_pct": self.max_position_size_pct, "risk_per_trade_pct": self.risk_per_trade_pct,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_optimal_position_size_duration_ms", duration, tags={"status": "success"})
            return result
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size for {symbol}: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_optimal_position_size_duration_ms", duration, tags={"status": "failed"})
            return {"error": str(e)}


    def calculate_portfolio_impact(
        self, symbol: str, position_size: float
    ) -> Dict[str, Any]:
        """
        Calculate the impact of a new position on the portfolio using RiskAnalysisMCP.
        (Code unchanged)
        """
        start_time = time.time()
        try:
            portfolio_data = self.portfolio_analyzer.get_portfolio_data()
            correlation_data = self.portfolio_analyzer.get_correlation_data()
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool(
                "calculate_portfolio_impact",
                {
                    "symbol": symbol, "position_size": position_size,
                    "portfolio_data": portfolio_data, "correlation_data": correlation_data,
                }
            )
            if result and result.get("error"): self.mcp_tool_error_count += 1
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_portfolio_impact_duration_ms", duration, tags={"status": "success"})
            return result
        except Exception as e:
            self.logger.error(f"Error calculating portfolio impact for {symbol}: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("decision_model.calculate_portfolio_impact_duration_ms", duration, tags={"status": "failed"})
            return {"error": str(e)}


    def store_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Store a trading decision in Redis.

        Args:
            decision: Dictionary containing the trading decision

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Storing decision for {decision.get('symbol')}: {decision.get('action')}")
        try:
            # Add timestamp if missing
            if "timestamp" not in decision:
                decision["timestamp"] = datetime.now().isoformat()

            # Generate a unique ID for the decision
            decision_id = str(uuid.uuid4())
            decision["decision_id"] = decision_id

            # Store the decision in Redis (e.g., in a sorted set for history and a separate key for latest per symbol)
            self.mcp_tool_call_count += 1
            # Store in sorted set for history (score by timestamp)
            history_result = self.redis_mcp.call_tool(
                "zadd",
                {
                    "key": self.redis_keys["decision_history"],
                    "score": int(datetime.fromisoformat(decision["timestamp"]).timestamp()),
                    "member": json.dumps(decision)
                }
            )
            if history_result and history_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to store decision history in Redis: {history_result.get('error')}")
                 # Continue, but log error

            # Store as latest decision per symbol
            symbol = decision.get("symbol")
            if symbol:
                self.mcp_tool_call_count += 1
                latest_result = self.redis_mcp.call_tool(
                    "set_json",
                    {
                        "key": f"{self.redis_keys['latest_decision']}{symbol}",
                        "value": decision,
                        "expiry": 86400 # Store latest decision for 1 day
                    }
                )
                if latest_result and latest_result.get("error"):
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Failed to store latest decision for {symbol} in Redis: {latest_result.get('error')}")
                     # Continue, but log error


            # Store in a list for recent decisions (optional, could use sorted set)
            # self.redis_mcp.call_tool("lpush", {"key": self.redis_keys["decisions"], "value": json.dumps(decision)})
            # self.redis_mcp.call_tool("ltrim", {"key": self.redis_keys["decisions"], "start": 0, "stop": 99}) # Keep only latest 100

            return True # Return True even if some storage failed, as long as no exception

        except Exception as e:
            self.logger.error(f"Error storing decision in Redis: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return False

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trading decisions from Redis.

        Args:
            limit: Maximum number of recent decisions to retrieve

        Returns:
            List of recent trading decisions
        """
        self.logger.info(f"Getting {limit} recent decisions from Redis history...")
        try:
            # Retrieve recent decisions from the sorted set (latest by timestamp)
            self.mcp_tool_call_count += 1
            results = self.redis_mcp.call_tool(
                "zrevrange",
                {
                    "key": self.redis_keys["decision_history"],
                    "start": 0,
                    "stop": limit - 1
                }
            )

            if results and not results.get("error"):
                decisions_raw = results.get("members", [])
                decisions = [json.loads(d) for d in decisions_raw]
                self.logger.info(f"Retrieved {len(decisions)} recent decisions.")
                return decisions
            elif results and results.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to get recent decisions from Redis: {results.get('error')}")
                 return []
            else:
                 self.logger.warning("No recent decisions found in Redis history.")
                 return []

        except Exception as e:
            self.logger.error(f"Error getting recent decisions from Redis: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return []

    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get data from the Selection Model via Redis.

        Returns:
            Selection Model data (or empty dict/error if not available)
        """
        self.logger.info("Getting selection data from Redis...")
        try:
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool("get_json", {"key": self.redis_keys["selection_data"]})
            if result and not result.get("error"):
                self.logger.info("Retrieved selection data.")
                return result.get("value", {}) if result else {}
            elif result and result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to get selection data from Redis: {result.get('error')}")
                 return {"error": result.get('error', 'Failed to get selection data from Redis')}
            else:
                 self.logger.warning("No selection data found in Redis.")
                 return {"message": "No selection data found in Redis."}

        except Exception as e:
            self.logger.error(f"Error getting selection data from Redis: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return {"error": str(e)}

    def get_finnlp_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get sentiment analysis data from the FinNLP Model via Redis.

        Args:
            symbol: Optional symbol to filter data for. If None, get overall market sentiment.

        Returns:
            FinNLP/sentiment analysis data (or empty dict/error if not available)
        """
        self.logger.info(f"Getting sentiment analysis data for symbol: {symbol if symbol else 'market overall'}")
        try:
            # Start timing
            start_time = time.time()
            
            if symbol:
                # Get specific symbol sentiment data
                self.mcp_tool_call_count += 1
                sentiment_key = f"{self.redis_keys['sentiment_data_key']}{symbol}"
                result = self.redis_mcp.call_tool("get_json", {"key": sentiment_key})
                
                if result and not result.get("error") and result.get("value"):
                    sentiment_data = result.get("value", {})
                    self.logger.info(f"Retrieved sentiment data for {symbol}: score={sentiment_data.get('sentiment_score', 'N/A')}")
                    
                    # Check if data is stale (older than 24 hours)
                    timestamp = sentiment_data.get("timestamp")
                    if timestamp:
                        data_time = datetime.fromisoformat(timestamp)
                        now = datetime.now()
                        if (now - data_time).total_seconds() > 86400:  # 24 hours
                            self.logger.warning(f"Sentiment data for {symbol} is stale ({timestamp})")
                            sentiment_data["is_stale"] = True
                    
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "success"})
                    return sentiment_data
                
                elif result and result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to get sentiment data for {symbol}: {result.get('error')}")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "error"})
                    return {"error": result.get('error')}
                
                else:
                    # Try to get from stream (recent updates)
                    self.mcp_tool_call_count += 1
                    stream_result = self.redis_mcp.call_tool(
                        "xrevrange", 
                        {
                            "stream": self.redis_keys['sentiment_reports_stream'],
                            "count": 20  # Check last 20 entries
                        }
                    )
                    
                    if stream_result and not stream_result.get("error") and stream_result.get("messages"):
                        for msg_id, msg_data in stream_result.get("messages", []):
                            # Check if this message contains data for our symbol
                            if msg_data.get("symbol") == symbol:
                                self.logger.info(f"Found sentiment data for {symbol} in stream")
                                duration = (time.time() - start_time) * 1000
                                self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "success_from_stream"})
                                return msg_data
                    
                    self.logger.warning(f"No sentiment data found for {symbol}")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "not_found"})
                    return {"message": f"No sentiment data found for {symbol}", "sentiment_score": 0.0, "is_stale": True}
            
            else:
                # Get overall market sentiment
                self.mcp_tool_call_count += 1
                market_sentiment_key = self.redis_keys["sentiment_market_key"]
                result = self.redis_mcp.call_tool("get_json", {"key": market_sentiment_key})
                
                if result and not result.get("error") and result.get("value"):
                    market_sentiment = result.get("value", {})
                    self.logger.info(f"Retrieved market sentiment: {market_sentiment.get('market_sentiment', 'N/A')}")
                    
                    # Check if data is stale
                    timestamp = market_sentiment.get("timestamp")
                    if timestamp:
                        data_time = datetime.fromisoformat(timestamp)
                        now = datetime.now()
                        if (now - data_time).total_seconds() > 86400:  # 24 hours
                            self.logger.warning(f"Market sentiment data is stale ({timestamp})")
                            market_sentiment["is_stale"] = True
                    
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "success"})
                    return market_sentiment
                
                elif result and result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to get market sentiment: {result.get('error')}")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "error"})
                    return {"error": result.get('error')}
                
                else:
                    # Try to get latest entry from the sentiment stream
                    self.mcp_tool_call_count += 1
                    latest_stream_result = self.redis_mcp.call_tool(
                        "xrevrange", 
                        {
                            "stream": self.redis_keys['sentiment_reports_stream'],
                            "count": 1  # Just get the most recent entry
                        }
                    )
                    
                    if latest_stream_result and not latest_stream_result.get("error") and latest_stream_result.get("messages"):
                        # The most recent entry might have market sentiment
                        msg_id, msg_data = latest_stream_result.get("messages")[0]
                        if msg_data.get("market_sentiment"):
                            self.logger.info("Found market sentiment in stream")
                            duration = (time.time() - start_time) * 1000
                            self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "success_from_stream"})
                            return msg_data
                    
                    self.logger.warning("No market sentiment data found")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_finnlp_data_duration_ms", duration, tags={"status": "not_found"})
                    return {"message": "No market sentiment data found", "market_sentiment": "neutral", "is_stale": True}

        except Exception as e:
            self.logger.error(f"Error getting sentiment data: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return {"error": str(e), "sentiment_score": 0.0, "is_stale": True}

    def get_forecaster_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data from the Forecaster Model via Redis.

        Args:
            symbol: Optional symbol to filter data for. If None, get overall data.

        Returns:
            Forecaster Model data (or empty dict/error if not available)
        """
        self.logger.info(f"Getting Forecaster data from Redis for symbol: {symbol if symbol else 'all'}")
        try:
            # This assumes Forecaster Model publishes data to a specific key or stream
            # Example: Get latest forecast data for a symbol
            if symbol:
                self.mcp_tool_call_count += 1
                # Assuming Forecaster Model stores latest forecast per symbol in a key like "forecaster:forecast:SYMBOL"
                forecast_key = f"forecaster:forecast:{symbol}"
                result = self.redis_mcp.call_tool("get_json", {"key": forecast_key})
                if result and not result.get("error"):
                    self.logger.info(f"Retrieved Forecaster data for {symbol}.")
                    return result.get("value", {}) if result else {}
                elif result and result.get("error"):
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Failed to get Forecaster data for {symbol} from Redis: {result.get('error')}")
                     return {"error": result.get('error', f'Failed to get Forecaster data for {symbol} from Redis')}
                else:
                     self.logger.warning(f"No Forecaster data found in Redis for {symbol}.")
                     return {"message": f"No Forecaster data found in Redis for {symbol}."}
            else:
                # Example: Get overall market forecast data (if stored in a single key)
                self.mcp_tool_call_count += 1
                # Assuming Forecaster Model stores overall market forecast in a key like "forecaster:market_forecast"
                overall_forecast_key = "forecaster:market_forecast"
                result = self.redis_mcp.call_tool("get_json", {"key": overall_forecast_key})
                if result and not result.get("error"):
                    self.logger.info("Retrieved overall Forecaster data.")
                    return result.get("value", {}) if result else {}
                elif result and result.get("error"):
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Failed to get overall Forecaster data from Redis: {result.get('error')}")
                     return {"error": result.get('error', 'Failed to get overall Forecaster data from Redis')}
                else:
                     self.logger.warning("No overall Forecaster data found in Redis.")
                     return {"message": "No overall Forecaster data found in Redis."}

        except Exception as e:
            self.logger.error(f"Error getting Forecaster data from Redis: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return {"error": str(e)}

    def get_fundamental_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get fundamental analysis data from Redis.

        Args:
            symbol: Optional symbol to filter data for. If None, get overall market fundamental data.

        Returns:
            Fundamental Analysis Model data with comprehensive financials and ratings
        """
        self.logger.info(f"Getting fundamental analysis data for {symbol if symbol else 'market overall'}")
        try:
            # Start timing
            start_time = time.time()
            
            if symbol:
                # Get fundamental data for specific symbol
                self.mcp_tool_call_count += 1
                fundamental_key = f"{self.redis_keys['fundamental_data_key']}{symbol}"
                result = self.redis_mcp.call_tool("get_json", {"key": fundamental_key})
                
                if result and not result.get("error") and result.get("value"):
                    fundamental_data = result.get("value", {})
                    self.logger.info(f"Retrieved fundamental data for {symbol}: score={fundamental_data.get('overall_rating', 'N/A')}")
                    
                    # Check if data is stale (fundamental data can be valid for longer, up to 7 days)
                    timestamp = fundamental_data.get("timestamp")
                    if timestamp:
                        data_time = datetime.fromisoformat(timestamp)
                        now = datetime.now()
                        if (now - data_time).total_seconds() > 604800:  # 7 days
                            self.logger.warning(f"Fundamental data for {symbol} is stale ({timestamp})")
                            fundamental_data["is_stale"] = True
                    
                    # Add validity context to help with decision making
                    if "financials" in fundamental_data:
                        latest_quarter = fundamental_data.get("financials", {}).get("latest_quarter")
                        if latest_quarter:
                            # Parse quarter date
                            try:
                                quarter_date = datetime.strptime(latest_quarter, "%Y-%m-%d")
                                days_since_report = (datetime.now() - quarter_date).days
                                fundamental_data["days_since_report"] = days_since_report
                                if days_since_report > 90:  # More than a quarter old
                                    fundamental_data["data_freshness"] = "outdated"
                                elif days_since_report > 45:  # Month and a half old
                                    fundamental_data["data_freshness"] = "aging"
                                else:
                                    fundamental_data["data_freshness"] = "recent"
                            except ValueError:
                                fundamental_data["data_freshness"] = "unknown"
                    
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "success"})
                    return fundamental_data
                
                elif result and result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to get fundamental data for {symbol}: {result.get('error')}")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "error"})
                    return {"error": result.get('error')}
                
                else:
                    # Try to get from stream (recent updates - fundamental data might be published on a schedule)
                    self.mcp_tool_call_count += 1
                    stream_result = self.redis_mcp.call_tool(
                        "xrevrange", 
                        {
                            "stream": self.redis_keys['fundamental_reports_stream'],
                            "count": 100  # Check more entries as fundamentals are less frequent
                        }
                    )
                    
                    if stream_result and not stream_result.get("error") and stream_result.get("messages"):
                        for msg_id, msg_data in stream_result.get("messages", []):
                            # Check if this message contains data for our symbol
                            if msg_data.get("symbol") == symbol:
                                self.logger.info(f"Found fundamental data for {symbol} in stream")
                                duration = (time.time() - start_time) * 1000
                                self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "success_from_stream"})
                                
                                # Store in Redis for future faster access
                                self.redis_mcp.call_tool(
                                    "set_json", 
                                    {
                                        "key": fundamental_key,
                                        "value": msg_data,
                                        "expiry": 86400  # Cache for 1 day
                                    }
                                )
                                
                                return msg_data
                    
                    self.logger.warning(f"No fundamental data found for {symbol}")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "not_found"})
                    return {
                        "message": f"No fundamental data found for {symbol}", 
                        "overall_rating": "unknown", 
                        "is_stale": True,
                        "data_freshness": "missing"
                    }
            
            else:
                # Get overall market fundamental analysis
                self.mcp_tool_call_count += 1
                market_fundamental_key = self.redis_keys["fundamental_market_key"]
                result = self.redis_mcp.call_tool("get_json", {"key": market_fundamental_key})
                
                if result and not result.get("error") and result.get("value"):
                    market_fundamentals = result.get("value", {})
                    self.logger.info(f"Retrieved market fundamental data: {market_fundamentals.get('market_outlook', 'N/A')}")
                    
                    # Check if data is stale
                    timestamp = market_fundamentals.get("timestamp")
                    if timestamp:
                        data_time = datetime.fromisoformat(timestamp)
                        now = datetime.now()
                        if (now - data_time).total_seconds() > 86400:  # 1 day
                            self.logger.warning(f"Market fundamental data is stale ({timestamp})")
                            market_fundamentals["is_stale"] = True
                    
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "success"})
                    return market_fundamentals
                
                elif result and result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to get market fundamental data: {result.get('error')}")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "error"})
                    return {"error": result.get('error')}
                
                else:
                    # Try to get latest entry from the fundamental reports stream that has market data
                    self.mcp_tool_call_count += 1
                    latest_stream_result = self.redis_mcp.call_tool(
                        "xrevrange", 
                        {
                            "stream": self.redis_keys['fundamental_reports_stream'],
                            "count": 20
                        }
                    )
                    
                    if latest_stream_result and not latest_stream_result.get("error") and latest_stream_result.get("messages"):
                        for msg_id, msg_data in latest_stream_result.get("messages", []):
                            if msg_data.get("market_outlook") or msg_data.get("sector_analysis"):
                                self.logger.info("Found market fundamental data in stream")
                                duration = (time.time() - start_time) * 1000
                                self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "success_from_stream"})
                                
                                # Store in Redis for future faster access
                                self.redis_mcp.call_tool(
                                    "set_json", 
                                    {
                                        "key": market_fundamental_key,
                                        "value": msg_data,
                                        "expiry": 86400  # Cache for 1 day
                                    }
                                )
                                
                                return msg_data
                    
                    self.logger.warning("No market fundamental data found")
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("decision_model.get_fundamental_data_duration_ms", duration, tags={"status": "not_found"})
                    return {
                        "message": "No market fundamental data found", 
                        "market_outlook": "neutral", 
                        "is_stale": True
                    }

        except Exception as e:
            self.logger.error(f"Error getting fundamental data: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return {"error": str(e), "overall_rating": "unknown", "is_stale": True}


    def handle_position_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle position updates (trades executed, positions closed) received via Redis Stream.

        Args:
            update_data: Dictionary containing the position update data

        Returns:
            Status of the update handling
        """
        self.logger.info(f"Handling position update: {update_data.get('event_type')}")
        try:
            event_type = update_data.get("event_type")
            if event_type == "positions_updated":
                positions = update_data.get("positions", [])
                self.logger.info(f"Received updated positions: {len(positions)}")
                # Store updated positions in Redis for PortfolioAnalyzer to use
                self.mcp_tool_call_count += 1
                store_result = self.redis_mcp.call_tool("set_json", {"key": self.redis_keys["positions"], "value": positions, "expiry": 60}) # Cache for 60 seconds
                if store_result and store_result.get("error"):
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Failed to store updated positions in Redis: {store_result.get('error')}")
                     return {"status": "error", "message": "Failed to store updated positions"}
                return {"status": "success", "message": "Positions updated"}

            elif event_type == "account_info_updated":
                 account_info = update_data.get("account_info", {})
                 self.logger.info(f"Received updated account info: {account_info.get('equity')}")
                 # Store updated account info in Redis
                 self.mcp_tool_call_count += 1
                 store_result = self.redis_mcp.call_tool("set_json", {"key": self.redis_keys["account_info"], "value": account_info, "expiry": 60}) # Cache for 60 seconds
                 if store_result and store_result.get("error"):
                      self.mcp_tool_error_count += 1
                      self.logger.error(f"Failed to store updated account info in Redis: {store_result.get('error')}")
                      return {"status": "error", "message": "Failed to store updated account info"}
                 return {"status": "success", "message": "Account info updated"}

            elif event_type == "capital_available":
                 available_capital = update_data.get("available_capital")
                 self.logger.info(f"Received available capital update: {available_capital}")
                 # Store available capital in Redis
                 self.mcp_tool_call_count += 1
                 store_result = self.redis_mcp.call_tool("set", {"key": self.redis_keys["capital_available"], "value": str(available_capital)})
                 if store_result and store_result.get("error"):
                      self.mcp_tool_error_count += 1
                      self.logger.error(f"Failed to store available capital in Redis: {store_result.get('error')}")
                      return {"status": "error", "message": "Failed to store available capital"}
                 return {"status": "success", "message": "Available capital updated"}

            else:
                self.logger.warning(f"Received unhandled position update event type: {event_type}")
                return {"status": "warning", "message": f"Unhandled event type: {event_type}"}

        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return {"status": "error", "error": str(e)}


    def request_new_selections(self, available_capital: float) -> Dict[str, Any]:
        """
        Request new stock selections when capital becomes available using Redis Stream.

        Args:
            available_capital: Amount of available capital for new positions

        Returns:
            Status of the request
        """
        self.logger.info(f"Requesting new selections with available capital: ${available_capital:.2f}")
        try:
            # Publish a request to the Selection Model's stream
            self.mcp_tool_call_count += 1
            request_id = str(uuid.uuid4())
            result = self.redis_mcp.call_tool(
                "xadd", # Assuming xadd tool exists for streams
                {
                    "stream": "model:selection:requests", # Example stream name for selection requests
                    "data": {"request_id": request_id, "available_capital": available_capital, "timestamp": datetime.now().isoformat()}
                }
            )
            if result and not result.get("error"):
                self.logger.info(f"Published selection request with ID: {request_id}")
                return {"status": "success", "request_id": request_id}
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to publish selection request to Redis stream: {result.get('error') if result else 'Unknown error'}")
                return {"status": "error", "error": result.get('error', 'Failed to publish selection request')}

        except Exception as e:
            self.logger.error(f"Error requesting new selections: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return {"status": "error", "error": str(e)}


    def check_selection_responses(self) -> List[Dict[str, Any]]:
        """
        Check for new selection responses from the Selection Model using Redis Stream.

        Returns:
            List of new selection responses
        """
        self.logger.info("Checking for new selection responses...")
        try:
            # Set up proper stream monitoring with consumer group if it doesn't exist
            stream_name = self.redis_keys.get("selection_reports_stream", "model:selection:candidates")
            group_name = "decision_model_consumers"
            consumer_name = f"decision_model_{os.getpid()}"  # Use process ID for unique consumer name
            
            # Try to create consumer group if it doesn't exist
            try:
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool(
                    "xgroup_create", 
                    {
                        "stream": stream_name,
                        "group": group_name,
                        "id": "$",  # Start from newest message
                        "mkstream": True  # Create stream if it doesn't exist
                    }
                )
                self.logger.info(f"Created consumer group '{group_name}' for stream '{stream_name}'")
            except Exception as e:
                # Group might already exist, which is fine
                if "BUSYGROUP" not in str(e):
                    self.logger.warning(f"Error creating consumer group: {e}")
            
            # Read new messages from the stream using consumer group
            self.mcp_tool_call_count += 1
            read_result = self.redis_mcp.call_tool(
                "xreadgroup", 
                {
                    "group": group_name,
                    "consumer": consumer_name,
                    "streams": [stream_name],
                    "ids": [">"],  # Read only new messages
                    "count": 10,  # Read up to 10 messages
                    "block": 1000  # Block for 1 second if no messages
                }
            )

            if read_result and not read_result.get("error"):
                messages = read_result.get("messages", [])
                responses = []
                if messages:
                    self.logger.info(f"Received {len(messages)} messages from selection responses stream.")
                    for stream_name, stream_messages in messages:
                        message_ids = []
                        for message_id, message_data in stream_messages:
                            self.logger.info(f"Processing selection response message {message_id} from {stream_name}")
                            
                            # Store the selection data in Redis for faster access
                            if message_data.get("candidates"):
                                self.mcp_tool_call_count += 1
                                self.redis_mcp.call_tool(
                                    "set_json", 
                                    {
                                        "key": self.redis_keys["selection_data_key"],
                                        "value": message_data,
                                        "expiry": 3600  # Cache for 1 hour
                                    }
                                )
                                self.logger.info(f"Stored selection data with {len(message_data.get('candidates', []))} candidates in Redis")
                            
                            responses.append(message_data)
                            message_ids.append(message_id)
                        
                        # Acknowledge processed messages
                        if message_ids:
                            self.mcp_tool_call_count += 1
                            self.redis_mcp.call_tool(
                                "xack", 
                                {
                                    "stream": stream_name,
                                    "group": group_name,
                                    "ids": message_ids
                                }
                            )
                            self.logger.info(f"Acknowledged {len(message_ids)} messages from stream {stream_name}")

                return responses

            elif read_result and read_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to read from selection responses stream: {read_result.get('error')}")
                 
                 # Fallback: try to read directly from the stream without consumer group
                 self.mcp_tool_call_count += 1
                 fallback_result = self.redis_mcp.call_tool(
                     "xread", 
                     {
                         "streams": [stream_name],
                         "count": 10,
                         "block": 100
                     }
                 )
                 
                 if fallback_result and not fallback_result.get("error"):
                     messages = fallback_result.get("messages", [])
                     responses = []
                     if messages:
                         self.logger.info(f"Fallback: Received {len(messages)} messages from selection responses stream.")
                         for stream_name, stream_messages in messages:
                             for message_id, message_data in stream_messages:
                                 responses.append(message_data)
                         return responses
                 
                 return []

            else:
                 self.logger.info("No new messages in selection responses stream.")
                 return []

        except Exception as e:
            self.logger.error(f"Error checking selection responses: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return []


    def _refresh_selection_data(self):
        """
        Refresh the selection data from Redis.
        (This method might be less needed if using streams for responses)
        """
        self.logger.warning("Attempted to refresh selection data via _refresh_selection_data. Consider using check_selection_responses.")
        pass # No-op as streams are preferred


    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Helper to get the latest price for a symbol using FinancialDataMCP."""
        try:
            self.mcp_tool_call_count += 1
            quote_result = self.financial_data_mcp.call_tool("get_latest_trade", {"symbol": symbol})
            if quote_result and not quote_result.get("error"):
                return quote_result.get("price")
            elif quote_result and quote_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to get latest price for {symbol}: {quote_result.get('error')}")
                 return None
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            self.execution_errors += 1
            self.logger.counter("decision_model.execution_errors")
            return None


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
