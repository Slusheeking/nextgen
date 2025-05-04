#!/usr/bin/env python3
"""
Test script for the AutoGen Orchestrator with OpenRouter integration

This script demonstrates how to instantiate and use the AutoGen orchestrator
with multiple LLMs through OpenRouter.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Make sure we can import modules from the project
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv(dotenv_path=project_root / ".env")
if not os.environ.get("OPENROUTER_API_KEY"):
    print("Warning: OPENROUTER_API_KEY not found in environment. Using default key.")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_orchestrator")

# Import the AutoGenOrchestrator
from fingpt.autogen_orchestrator.autogen_model import AutoGenOrchestrator, create_orchestrator


def test_orchestrator(mock_run=True):
    """
    Test the basic functionality of the AutoGen orchestrator with OpenRouter.
    
    Args:
        mock_run: If True, run in test mode without making actual API calls
    """
    
    logger.info("Creating AutoGen orchestrator with OpenRouter LLMs...")
    orchestrator = create_orchestrator()
    
    # Sample market data for testing
    sample_market_data = {
        "sp500": 5024.35,
        "vix": 14.87,
        "treasury_10y": 4.12,
        "market_sentiment": "mixed"
    }
    
    if mock_run:
        logger.info("*** RUNNING IN TEST MODE - NO ACTUAL API CALLS WILL BE MADE ***")
        logger.info("To make actual API calls, set mock_run=False")
        logger.info("This will use your OpenRouter API key and may incur costs.")
    else:
        logger.info("*** RUNNING WITH REAL API CALLS - THIS WILL USE YOUR API KEY AND MAY INCUR COSTS ***")
    
    # Run a test trading cycle
    logger.info("Running test trading cycle with sample market data...")
    if not mock_run:
        logger.info("This may take a minute as it needs to communicate with the LLMs via OpenRouter...")
    result = orchestrator.run_trading_cycle(sample_market_data, mock=mock_run)
    
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
    logger.info("Starting OpenRouter AutoGen orchestrator test")
    logger.info("Using models: Claude 3 Opus, Llama 3, Gemini")
    test_orchestrator(mock_run=False)  # Set to False to make actual API calls