"""
Decision Analytics MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
analytics capabilities for making trading decisions by aggregating data from
multiple analysis models and calculating confidence scores.
"""

import os
import json
import numpy as np
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-decision-analytics",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class DecisionAnalyticsMCP(BaseMCPServer):
    """
    MCP server for aggregating analysis data and calculating decision confidence.

    This tool helps the Decision Model by synthesizing inputs from sentiment,
    technical, fundamental, risk, and context models to produce a confidence
    score for potential trading actions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Decision Analytics MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - weighting_model_path: Path to a model for dynamic weighting
                - confidence_thresholds: Thresholds for buy/sell/hold signals
                - feature_weights: Static weights for different analysis types
                - cache_dir: Directory for caching intermediate results
        """
        super().__init__(name="decision_analytics_mcp", config=config)

        # Set default configurations
        self.weighting_model_path = self.config.get("weighting_model_path", None)
        self.confidence_thresholds = self.config.get(
            "confidence_thresholds",
            {"buy": 0.7, "sell": 0.7, "hold_min": 0.3, "hold_max": 0.7},
        )
        self.feature_weights_path = self.config.get(
            "feature_weights_path",
            os.path.join(
                os.path.dirname(__file__), "data/decision_feature_weights.json"
            ),
        )
        self.cache_dir = self.config.get("cache_dir", "./decision_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load weights and models
        self._load_feature_weights()
        self._load_weighting_model()

        # Register tools
        self._register_tools()

    def _load_feature_weights(self):
        """Load static weights for different analysis features."""
        self.feature_weights = {}

        if os.path.exists(self.feature_weights_path):
            try:
                with open(self.feature_weights_path, "r") as f:
                    self.feature_weights = json.load(f)
                self.logger.info(
                    f"Loaded feature weights from {self.feature_weights_path}"
                )
            except Exception as e:
                self.logger.error(f"Error loading feature weights: {e}")
                self._create_default_weights()
        else:
            self.logger.warning(
                f"Feature weights file not found: {self.feature_weights_path}"
            )
            self._create_default_weights()

    def _create_default_weights(self):
        """Create default weights for analysis features."""
        self.feature_weights = {
            "sentiment": {
                "overall_score": 0.6,
                "news_volume_spike": 0.2,
                "social_media_trend": 0.2,
            },
            "technical": {
                "trend_strength": 0.3,  # e.g., ADX
                "momentum_signal": 0.3,  # e.g., RSI, MACD crossover
                "pattern_confirmation": 0.25,  # e.g., Breakout from triangle
                "support_resistance_proximity": 0.15,
            },
            "fundamental": {
                "overall_health_score": 0.4,
                "valuation_relative_sector": 0.3,
                "growth_potential": 0.3,
            },
            "risk": {
                "volatility_adjusted_return": 0.4,  # e.g., Sharpe ratio proxy
                "max_drawdown_potential": 0.3,
                "correlation_with_portfolio": 0.3,  # Lower is better
            },
            "context": {
                "historical_pattern_success_rate": 0.5,
                "analogous_event_outcome": 0.5,
            },
            "overall_category_weights": {
                "sentiment": 0.15,
                "technical": 0.30,
                "fundamental": 0.25,
                "risk": 0.20,
                "context": 0.10,
            },
        }
        self.logger.info("Created default feature weights")

    def _load_weighting_model(self):
        """Load a model for dynamic weighting if specified."""
        self.weighting_model = None
        if self.weighting_model_path and os.path.exists(self.weighting_model_path):
            try:
                #
                # In a real implementation, load a ML model (e.g.,
                # scikit-learn, tensorflow)
                #
                # Example: self.weighting_model =
                # joblib.load(self.weighting_model_path)
                self.logger.info(
                    f"Dynamically weighting model would be loaded from {self.weighting_model_path} (placeholder)"
                )
                # Placeholder: Set a flag or dummy model
                self.weighting_model = "dynamic_model_loaded"
            except Exception as e:
                self.logger.error(f"Error loading weighting model: {e}")
        else:
            self.logger.info("Using static feature weights for decision scoring.")

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "aggregate_analysis_data",
            self.aggregate_analysis_data,
            "Aggregate analysis data from multiple sources for a specific stock",
            {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to aggregate data for",
                },
                "analysis_inputs": {
                    "type": "object",
                    "description": "Dictionary containing results from various analysis models (sentiment, technical, fundamental, risk, context)",
                },
            },
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "aggregated_data": {"type": "object"},
                    "summary": {"type": "string"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_confidence_score",
            self.calculate_confidence_score,
            "Calculate a confidence score for a potential trading action (buy/sell/hold)",
            {
                "symbol": {"type": "string", "description": "Stock symbol"},
                "action": {
                    "type": "string",
                    "description": "Potential action: 'buy', 'sell', or 'hold'",
                },
                "aggregated_data": {
                    "type": "object",
                    "description": "Aggregated analysis data for the stock",
                },
                "market_context": {
                    "type": "object",
                    "description": "Current overall market context (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string"},
                    "confidence_score": {"type": "number"},  # Score between 0.0 and 1.0
                    "signal_strength": {
                        "type": "string"
                    },  # e.g., 'strong_buy', 'weak_sell'
                    "contributing_factors": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "explain_decision_score",
            self.explain_decision_score,
            "Provide an explanation for a calculated confidence score",
            {
                "symbol": {"type": "string", "description": "Stock symbol"},
                "action": {
                    "type": "string",
                    "description": "Action considered (buy/sell/hold)",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The calculated confidence score",
                },
                "contributing_factors": {
                    "type": "object",
                    "description": "Factors contributing to the score",
                },
            },
            {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "key_drivers": {"type": "array"},
                    "counter_signals": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

    def aggregate_analysis_data(
        self, symbol: str, analysis_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate analysis data from multiple sources for a specific stock.

        Args:
            symbol: Stock symbol.
            analysis_inputs: Dictionary containing results from various models.
                             Example: {'sentiment': {...}, 'technical': {...}, ...}

        Returns:
            Dictionary with aggregated data and a summary.
        """
        start_time = time.time()

        aggregated_data = {}
        summary_points = []

        # Process Sentiment Data
        sentiment_data = analysis_inputs.get("sentiment", {})
        if sentiment_data:
            agg_sentiment = {
                "overall_score": sentiment_data.get(
                    "overall_score", 0.5
                ),  # Normalize to 0-1? Assume 0=neg, 0.5=neu, 1=pos
                "confidence": sentiment_data.get("confidence", 0.0),
                "positive_ratio": sentiment_data.get("positive_ratio", 0.0),
                "negative_ratio": sentiment_data.get("negative_ratio", 0.0),
                "news_volume_spike": sentiment_data.get(
                    "news_volume_spike", False
                ),  # Example derived metric
                "social_trend": sentiment_data.get(
                    "social_trend", "neutral"
                ),  # Example derived metric
            }
            aggregated_data["sentiment"] = agg_sentiment
            summary_points.append(
                f"Sentiment score: {agg_sentiment['overall_score']:.2f} (Conf: {agg_sentiment['confidence']:.2f})"
            )
            if agg_sentiment["news_volume_spike"]:
                summary_points.append("High news volume detected.")

        # Process Technical Data
        technical_data = analysis_inputs.get("technical", {})
        if technical_data:
            # Extract key signals and indicators
            signals = technical_data.get("signals", [])
            indicators = technical_data.get("indicators", {})
            patterns = analysis_inputs.get("patterns", {}).get(
                "patterns", []
            )  # Get patterns if available

            # Example: Extract trend strength (e.g., from ADX)
            adx_value = indicators.get("adx", {}).get("adx", [np.nan])[-1]
            trend_strength = 0.5  # Default neutral
            if not np.isnan(adx_value):
                trend_strength = min(
                    1.0, adx_value / 50.0
                )  # Normalize ADX (e.g., 50 is strong)

            # Example: Check for recent bullish signals
            recent_bullish_signals = [
                s for s in signals if "bullish" in s.get("type", "").lower()
            ][-3:]  # Last 3 bullish
            recent_bearish_signals = [
                s for s in signals if "bearish" in s.get("type", "").lower()
            ][-3:]  # Last 3 bearish

            # Example: Check for confirmed patterns
            confirmed_patterns = [p for p in patterns if p.get("confidence", 0) > 0.7]

            agg_technical = {
                "trend_strength": trend_strength,
                "recent_bullish_signals": len(recent_bullish_signals),
                "recent_bearish_signals": len(recent_bearish_signals),
                "confirmed_patterns": [p.get("pattern") for p in confirmed_patterns],
                # Add more aggregated technical metrics
            }
            aggregated_data["technical"] = agg_technical
            summary_points.append(f"Technical trend strength: {trend_strength:.2f}")
            if confirmed_patterns:
                summary_points.append(
                    f"Confirmed patterns: {', '.join(agg_technical['confirmed_patterns'])}"
                )

        # Process Fundamental Data
        fundamental_data = analysis_inputs.get("fundamental", {})
        if fundamental_data:
            health_score = fundamental_data.get("score", {}).get("overall_score", 0.5)
            valuation_comparison = fundamental_data.get(
                "comparison", {}
            )  # Compare specific ratios

            # Example: Assess valuation relative to sector
            pe_comp = valuation_comparison.get("pe_ratio", {})
            pb_comp = valuation_comparison.get("pb_ratio", {})
            valuation_status = "neutral"
            if pe_comp.get("status") == "better" and pb_comp.get("status") == "better":
                valuation_status = "undervalued"
            elif pe_comp.get("status") == "worse" and pb_comp.get("status") == "worse":
                valuation_status = "overvalued"

            agg_fundamental = {
                "health_score": health_score,
                "valuation_status": valuation_status,
                # Add more aggregated fundamental metrics
            }
            aggregated_data["fundamental"] = agg_fundamental
            summary_points.append(f"Fundamental health score: {health_score:.2f}")
            summary_points.append(f"Valuation vs sector: {valuation_status}")

        # Process Risk Data
        risk_data = analysis_inputs.get("risk", {})
        if risk_data:
            # Example: Extract key risk metrics
            volatility = risk_data.get("volatility", {}).get("annualized", np.nan)
            sharpe_ratio = risk_data.get("performance", {}).get(
                "sharpe_ratio", np.nan
            )  # Example metric
            max_drawdown = risk_data.get("drawdown", {}).get("max_drawdown", np.nan)
            correlation = risk_data.get("correlation", {}).get(
                "avg_correlation_portfolio", 0.0
            )

            agg_risk = {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "portfolio_correlation": correlation,
            }
            aggregated_data["risk"] = agg_risk
            if not np.isnan(volatility):
                summary_points.append(f"Annualized Volatility: {volatility:.2%}")
            if not np.isnan(sharpe_ratio):
                summary_points.append(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Process Context Data
        context_data = analysis_inputs.get("context", {})
        if context_data:
            # Example: Extract historical pattern performance
            historical_perf = context_data.get("historical_pattern_performance", {})
            analogous_events = context_data.get("analogous_events", [])

            agg_context = {
                "historical_pattern_accuracy": historical_perf.get("accuracy", 0.5),
                "num_analogous_events": len(analogous_events),
                # Add more aggregated context metrics
            }
            aggregated_data["context"] = agg_context
            if historical_perf:
                summary_points.append(
                    f"Historical pattern accuracy: {historical_perf.get('accuracy', 0.5):.2f}"
                )

        # Create summary string
        summary = f"Aggregation for {symbol}: " + " | ".join(summary_points)

        return {
            "symbol": symbol,
            "aggregated_data": aggregated_data,
            "summary": summary,
            "processing_time": time.time() - start_time,
        }

    def calculate_confidence_score(
        self,
        symbol: str,
        action: str,
        aggregated_data: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate a confidence score for a potential trading action.

        Args:
            symbol: Stock symbol.
            action: Potential action ('buy', 'sell', 'hold').
            aggregated_data: Aggregated analysis data from `aggregate_analysis_data`.
            market_context: Optional overall market context.

        Returns:
            Dictionary with confidence score and contributing factors.
        """
        start_time = time.time()

        overall_score = 0.0
        total_weight = 0.0
        contributing_factors = {}

        # Get overall category weights
        category_weights = self.feature_weights.get("overall_category_weights", {})

        # Use dynamic weighting model if available
        if self.weighting_model:
            # Placeholder: In reality, use the ML model to get dynamic weights
            # based on market context and aggregated data features.
            # dynamic_weights = self.weighting_model.predict(features)
            self.logger.info("Dynamic weighting model would be used here (placeholder)")
            # For now, just adjust static weights slightly based on context
            if market_context and market_context.get("volatility_regime") == "high":
                category_weights["risk"] = (
                    category_weights.get("risk", 0.20) * 1.2
                )  # Increase risk weight
                category_weights["technical"] = (
                    category_weights.get("technical", 0.30) * 0.9
                )  # Decrease technical weight
            # Normalize weights again if adjusted
            current_total = sum(category_weights.values())
            if current_total > 0:
                category_weights = {
                    k: v / current_total for k, v in category_weights.items()
                }

        # --- Score each category based on the desired action ---
        #
        # The scoring logic here is simplified. A real system would be more
        # nusanced.

        # Sentiment Score (0-1, higher is more bullish)
        sentiment_score = 0.5  # Neutral default
        if "sentiment" in aggregated_data:
            sentiment_info = aggregated_data["sentiment"]
            # Example: Combine overall score and news spike
            base_score = sentiment_info.get("overall_score", 0.5)
            spike_boost = 0.1 if sentiment_info.get("news_volume_spike", False) else 0.0
            sentiment_score = np.clip(base_score + spike_boost, 0, 1)
            contributing_factors["sentiment"] = sentiment_score * category_weights.get(
                "sentiment", 0
            )

        # Technical Score (0-1, higher is more bullish)
        technical_score = 0.5  # Neutral default
        if "technical" in aggregated_data:
            tech_info = aggregated_data["technical"]
            # Example: Combine trend, signals, patterns
            trend_score = tech_info.get("trend_strength", 0.5)
            signal_balance = (
                tech_info.get("recent_bullish_signals", 0)
                - tech_info.get("recent_bearish_signals", 0)
            ) / 3.0  # Max +/-1
            pattern_score = 0.5 + 0.2 * len(
                tech_info.get("confirmed_patterns", [])
            )  # Boost for patterns
            technical_score = np.clip(
                0.4 * trend_score
                + 0.4 * (0.5 + 0.5 * signal_balance)
                + 0.2 * pattern_score,
                0,
                1,
            )
            contributing_factors["technical"] = technical_score * category_weights.get(
                "technical", 0
            )

        # Fundamental Score (0-1, higher is fundamentally stronger)
        fundamental_score = 0.5  # Neutral default
        if "fundamental" in aggregated_data:
            fund_info = aggregated_data["fundamental"]
            # Example: Combine health score and valuation
            health_score = fund_info.get("health_score", 0.5)
            valuation_mult = 1.0
            if fund_info.get("valuation_status") == "undervalued":
                valuation_mult = 1.1
            if fund_info.get("valuation_status") == "overvalued":
                valuation_mult = 0.9
            fundamental_score = np.clip(health_score * valuation_mult, 0, 1)
            contributing_factors["fundamental"] = (
                fundamental_score * category_weights.get("fundamental", 0)
            )

        # Risk Score (0-1, higher is LOWER risk)
        risk_score = 0.5  # Neutral default
        if "risk" in aggregated_data:
            risk_info = aggregated_data["risk"]
            #
            # Example: Combine volatility, sharpe, correlation (lower
            # correlation is better)
            vol_score = 1.0 - np.clip(
                risk_info.get("volatility", 0.3) / 0.6, 0, 1
            )  # Assuming 60% vol is max risk
            sharpe_score = np.clip(
                0.5 + risk_info.get("sharpe_ratio", 0) / 4.0, 0, 1
            )  # Assuming Sharpe of 2 is max score
            corr_score = 1.0 - np.clip(
                risk_info.get("portfolio_correlation", 0.5), 0, 1
            )
            risk_score = np.clip(
                0.4 * vol_score + 0.4 * sharpe_score + 0.2 * corr_score, 0, 1
            )
            contributing_factors["risk"] = risk_score * category_weights.get("risk", 0)

        # Context Score (0-1, higher suggests favorable historical context)
        context_score = 0.5  # Neutral default
        if "context" in aggregated_data:
            context_info = aggregated_data["context"]
            hist_acc = context_info.get("historical_pattern_accuracy", 0.5)
            context_score = np.clip(hist_acc, 0, 1)
            contributing_factors["context"] = context_score * category_weights.get(
                "context", 0
            )

        # --- Combine category scores based on action ---
        if action == "buy":
            #
            # For buy, we want high scores in sentiment, technical,
            # fundamental, context, and risk (low risk)
            overall_score = sum(contributing_factors.values())
            total_weight = sum(
                category_weights.get(cat, 0) for cat in contributing_factors.keys()
            )

        elif action == "sell":
            # For sell, we want low scores (or inverse scores)
            # Inverse scores: 1 - score
            overall_score = (
                (1.0 - sentiment_score) * category_weights.get("sentiment", 0)
                + (1.0 - technical_score) * category_weights.get("technical", 0)
                + (1.0 - fundamental_score) * category_weights.get("fundamental", 0)
                + (1.0 - risk_score)
                * category_weights.get("risk", 0)  # High risk contributes to sell
                + (1.0 - context_score)
                * category_weights.get("context", 0)  # Unfavorable history
            )
            total_weight = sum(category_weights.values())

        elif action == "hold":
            #
            # Hold might be indicated by scores near the middle or conflicting
            # signals
            # Calculate distance from 0.5 for each score
            dist_sentiment = abs(sentiment_score - 0.5)
            dist_technical = abs(technical_score - 0.5)
            dist_fundamental = abs(fundamental_score - 0.5)
            dist_risk = abs(risk_score - 0.5)
            dist_context = abs(context_score - 0.5)

            #
            # Average distance - lower distance means more neutral/conflicting
            # -> higher hold score
            avg_distance = np.mean(
                [
                    dist_sentiment,
                    dist_technical,
                    dist_fundamental,
                    dist_risk,
                    dist_context,
                ]
            )
            overall_score = 1.0 - (
                avg_distance / 0.5
            )  # Scale distance (0-0.5) to score (1-0)
            total_weight = 1.0  # Hold score is calculated differently

        else:
            self.logger.warning(f"Unknown action: {action}")
            overall_score = 0.0
            total_weight = 1.0

        # Normalize the final score
        if total_weight > 0:
            confidence_score = np.clip(overall_score / total_weight, 0, 1)
        else:
            confidence_score = 0.5  # Default neutral if no weights

        # Determine signal strength
        signal_strength = "neutral"
        thresh = self.confidence_thresholds
        if action == "buy":
            if confidence_score >= thresh["buy"]:
                signal_strength = "strong_buy"
            elif confidence_score >= (thresh["buy"] + thresh["hold_max"]) / 2:
                signal_strength = "weak_buy"
        elif action == "sell":
            if confidence_score >= thresh["sell"]:
                signal_strength = "strong_sell"
            elif confidence_score >= (thresh["sell"] + thresh["hold_max"]) / 2:
                signal_strength = "weak_sell"
        elif action == "hold":
            if confidence_score >= 0.7:
                signal_strength = "strong_hold"  # High confidence in hold
            elif confidence_score >= 0.5:
                signal_strength = "weak_hold"

        # If score is low for buy/sell, it might imply hold
        if action in ["buy", "sell"] and confidence_score < thresh["hold_min"]:
            signal_strength = "hold_implied"

        # Refine contributing factors to show score * weight
        final_factors = {k: float(v) for k, v in contributing_factors.items()}

        return {
            "symbol": symbol,
            "action": action,
            "confidence_score": float(confidence_score),
            "signal_strength": signal_strength,
            "contributing_factors": final_factors,
            "processing_time": time.time() - start_time,
        }

    def explain_decision_score(
        self,
        symbol: str,
        action: str,
        confidence_score: float,
        contributing_factors: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Provide an explanation for a calculated confidence score.

        Args:
            symbol: Stock symbol.
            action: Action considered.
            confidence_score: The calculated confidence score.
            contributing_factors: Factors contributing to the score.

        Returns:
            Dictionary with explanation, key drivers, and counter signals.
        """
        start_time = time.time()

        explanation = f"Explanation for {action} decision on {symbol} (Confidence: {confidence_score:.2f}):\n"

        # Sort factors by absolute contribution
        sorted_factors = sorted(
            contributing_factors.items(), key=lambda item: abs(item[1]), reverse=True
        )

        key_drivers = []
        counter_signals = []

        # Determine expected direction of score for the action
        positive_action = action == "buy"

        for factor, contribution in sorted_factors:
            # Determine if the factor supports or opposes the action
            # This logic assumes higher scores (closer to 1) support 'buy'
            # and lower scores (closer to 0) support 'sell' (or oppose 'buy')

            #
            # We need the original category score, not the weighted
            # contribution
            #
            # This requires passing more data or recalculating category scores
            # here.
            #
            # For simplicity, let's infer based on contribution (positive
            # contribution supports buy)

            supports_action = (
                (contribution > 0) if positive_action else (contribution < 0)
            )  # Simplified logic


            # Add more detail based on factor type (requires original scores)
            # Example: factor_explanation += f"(Score: {original_score:.2f})"

            if supports_action:
                explanation += f"  - Supporting factor: {factor.capitalize()} (Contribution: {contribution:.3f})\n"
                key_drivers.append(
                    {"factor": factor, "contribution": float(contribution)}
                )
            else:
                explanation += f"  - Counter signal: {factor.capitalize()} (Contribution: {contribution:.3f})\n"
                counter_signals.append(
                    {"factor": factor, "contribution": float(contribution)}
                )

        if not key_drivers:
            explanation += "No strong supporting factors identified.\n"
        if not counter_signals:
            explanation += "No significant counter signals identified.\n"

        return {
            "explanation": explanation.strip(),
            "key_drivers": key_drivers,
            "counter_signals": counter_signals,
            "processing_time": time.time() - start_time,
        }


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {}

    # Create and start the server
    server = DecisionAnalyticsMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("DecisionAnalyticsMCP server started")

    # Example usage
    sample_analysis_inputs = {
        "sentiment": {
            "overall_score": 0.8,
            "confidence": 0.9,
            "news_volume_spike": True,
        },
        "technical": {
            "indicators": {"adx": {"adx": [35.0]}},
            "signals": [{"type": "bullish_crossover", "indicator": "MACD"}],
            "patterns": [{"pattern": "ascending_triangle", "confidence": 0.8}],
        },
        "fundamental": {
            "score": {"overall_score": 0.7},
            "comparison": {"pe_ratio": {"status": "better"}},
        },
        "risk": {
            "volatility": {"annualized": 0.25},
            "performance": {"sharpe_ratio": 1.5},
            "correlation": {"avg_correlation_portfolio": 0.2},
        },
        "context": {"historical_pattern_performance": {"accuracy": 0.75}},
    }

    # Aggregate data
    agg_result = server.aggregate_analysis_data("AAPL", sample_analysis_inputs)
    print(f"Aggregated Data: {json.dumps(agg_result, indent=2)}")

    # Calculate confidence score for 'buy'
    confidence_result = server.calculate_confidence_score(
        "AAPL", "buy", agg_result["aggregated_data"]
    )
    print(f"\nConfidence Score: {json.dumps(confidence_result, indent=2)}")

    # Explain the score
    explanation_result = server.explain_decision_score(
        "AAPL",
        "buy",
        confidence_result["confidence_score"],
    confidence_result["contributing_factors"],
        )
    print(f"\nExplanation: {json.dumps(explanation_result, indent=2)}")
