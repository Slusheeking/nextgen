"""
NextGen FinGPT Monitoring Module

This module provides comprehensive monitoring capabilities for the NextGen FinGPT project,
including Prometheus metrics collection, file-based logging, and system metrics monitoring.
It exposes a clean API for frontend integration and dashboard connectivity through a FastAPI server.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# Export the main components from the system_monitor module
from monitoring.system_monitor import (
    MonitoringManager,
    SystemMetricsCollector
)

# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create a master log file if it doesn't exist
master_log = os.path.join(logs_dir, "master.log")
if not os.path.exists(master_log):
    with open(master_log, "w") as f:
        f.write(f"--- Master log created ---\n")

# Set up a root logger with file handlers
def setup_root_logger():
    """Set up the root logger with file handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Check if the root logger already has handlers
    if not root_logger.handlers:
        # Add a console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Add a file handler for the master log
        master_handler = RotatingFileHandler(master_log, maxBytes=10*1024*1024, backupCount=10)
        master_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        master_handler.setFormatter(master_formatter)
        root_logger.addHandler(master_handler)

# Set up the root logger
setup_root_logger()

# Export the setup_monitoring function for backward compatibility
def setup_monitoring(service_name, enable_prometheus=True, default_labels=None, **kwargs):
    """
    Set up monitoring for a service.
    
    Args:
        service_name: Name of the service being monitored
        enable_prometheus: Whether to enable Prometheus metrics
        default_labels: Default labels to apply to all metrics and logs
        
    Returns:
        Tuple of (MonitoringManager, metrics_dict)
    """
    # Check for deprecated parameters
    if 'enable_loki' in kwargs:
        logging.getLogger(__name__).warning(
            "The 'enable_loki' parameter is deprecated. "
            "Logs are now stored in files and can be accessed through the Log API."
        )
    
    # Create a monitoring manager
    monitor = MonitoringManager(
        service_name=service_name,
    )
    
    # Create a metrics dictionary for compatibility with existing code
    metrics = {
        "requests_total": "requests_total",
        "errors_total": "errors_total",
        "active_requests": "active_requests",
        "request_duration_seconds": "request_duration_seconds",
        "operations_total": "operations_total",
        "operation_duration_seconds": "operation_duration_seconds"
    }
    
    return monitor, metrics

# Define the version
__version__ = '3.0.0'

# Define what gets imported with "from monitoring import *"
__all__ = [
    'MonitoringManager',
    'SystemMetricsCollector',
    'setup_monitoring',
]
