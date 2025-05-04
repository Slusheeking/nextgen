#!/usr/bin/env python3
"""
System Metrics Collector Main Entry Point

This module serves as the entry point for running the system metrics collector
as a Python module (python -m monitoring.system_metrics).
"""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_metrics_main")

# Add the parent directory to the path to allow importing from monitoring
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the SystemMetricsService from the collector module
from monitoring.system_metrics_collector import SystemMetricsService

def main():
    """Main entry point for the system metrics collector."""
    logger.info("Starting System Metrics Collector from module entry point...")
    
    # Create and start the service
    service = SystemMetricsService()
    
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        service.stop()
        logger.info("System Metrics Collector service stopped")

if __name__ == "__main__":
    main()
