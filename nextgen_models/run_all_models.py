#!/usr/bin/env python3
"""
Unified Model Launcher

This script instantiates all main model classes in nextgen_models and keeps them running.
Adapt as needed if new models are added.
"""

import time

from nextgen_models.autogen_orchestrator.autogen_model import AutogenOrchestratorModel
from nextgen_models.nextgen_context_model.context_model import ContextModel
from nextgen_models.nextgen_decision.decision_model import DecisionModel
from nextgen_models.nextgen_fundamental_analysis.fundamental_analysis_model import FundamentalAnalysisModel
from nextgen_models.nextgen_market_analysis.market_analysis_model import MarketAnalysisModel
from nextgen_models.nextgen_risk_assessment.risk_assessment_model import RiskAssessmentModel
from nextgen_models.nextgen_select.select_model import SelectModel
from nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model import SentimentAnalysisModel
from nextgen_models.nextgen_trader.trade_model import TradeModel

def main():
    print("Starting all NextGen model servers...")
    models = [
        AutogenOrchestratorModel(),
        ContextModel(),
        DecisionModel(),
        FundamentalAnalysisModel(),
        MarketAnalysisModel(),
        RiskAssessmentModel(),
        SelectModel(),
        SentimentAnalysisModel(),
        TradeModel(),
    ]
    print(f"{len(models)} model servers instantiated and running.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Unified model launcher stopped.")

if __name__ == "__main__":
    main()