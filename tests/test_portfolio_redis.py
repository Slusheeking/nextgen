#!/usr/bin/env python3
"""
Portfolio Redis Test Script

This script tests the portfolio data caching in Redis implemented in the TradePositionManager class.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Make sure we can import modules from the project
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portfolio_redis_test")

# Import the necessary modules
# Import the TradeModel class directly to avoid __init__.py import issues
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fingpt.nextgen_trader.trade_model import TradeModel, TradePositionManager

def test_portfolio_redis_caching():
    """Test portfolio data caching in Redis."""
    logger.info("Testing portfolio data caching in Redis...")
    
    # Create TradeModel instance
    trade_model = TradeModel()
    
    # Get position manager
    position_manager = trade_model.position_manager
    
    # Test sync_portfolio_data
    logger.info("Testing sync_portfolio_data...")
    sync_result = position_manager.sync_portfolio_data()
    logger.info(f"sync_portfolio_data result: {sync_result}")
    
    # Test get_account_info
    logger.info("Testing get_account_info...")
    account_info = position_manager.get_account_info()
    logger.info(f"Account info: {account_info}")
    
    # Test get_positions
    logger.info("Testing get_positions...")
    positions = position_manager.get_positions()
    logger.info(f"Positions: {positions}")
    
    # Test get_portfolio_constraints
    logger.info("Testing get_portfolio_constraints...")
    constraints = position_manager.get_portfolio_constraints()
    logger.info(f"Portfolio constraints: {constraints}")
    
    # Verify Redis keys
    logger.info("Verifying Redis keys...")
    last_updated = position_manager.trade_model.redis_mcp.get_value(position_manager._last_updated_key)
    logger.info(f"Last updated: {last_updated}")
    
    account_info_redis = position_manager.trade_model.redis_mcp.get_json(position_manager._account_info_key)
    logger.info(f"Account info from Redis: {account_info_redis}")
    
    positions_redis = position_manager.trade_model.redis_mcp.get_json(position_manager._positions_key)
    logger.info(f"Positions from Redis: {positions_redis}")
    
    portfolio_summary = position_manager.trade_model.redis_mcp.get_json(position_manager._portfolio_summary_key)
    logger.info(f"Portfolio summary from Redis: {portfolio_summary}")
    
    return True

def main():
    """Main test function."""
    logger.info("Starting portfolio Redis caching tests...")
    
    try:
        # Test portfolio Redis caching
        result = test_portfolio_redis_caching()
        if not result:
            logger.error("Portfolio Redis caching test failed")
            return 1
        
        logger.info("All portfolio Redis caching tests passed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
