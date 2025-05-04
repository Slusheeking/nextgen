#!/usr/bin/env python3
"""
Prometheus and Loki Utilities

This module provides simplified utilities for integrating Prometheus metrics
and Loki logging into NextGen FinGPT components. It serves as a compatibility
layer for existing code that uses the older prometheus_loki_utils module.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

# Import from the consolidated monitoring module
from monitoring.monitoring import setup_monitoring as _setup_monitoring

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prometheus_loki_utils")

def setup_monitoring(
    service_name: str,
    enable_prometheus: bool = True,
    enable_loki: bool = True,
    default_labels: Optional[Dict[str, str]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Set up monitoring for a service.
    
    This function is a compatibility wrapper around the setup_monitoring function
    from the consolidated monitoring module. It maintains the same interface for
    existing code that uses this module.
    
    Args:
        service_name (str): Name of the service for metrics and logs identification.
        enable_prometheus (bool): Whether to enable Prometheus metrics. Defaults to True.
        enable_loki (bool): Whether to enable Loki logging. Defaults to True.
        default_labels (dict, optional): Default labels to apply to all metrics and logs.
    
    Returns:
        tuple: A tuple containing the MonitoringManager instance and a dictionary of metrics.
    """
    # Get configuration from environment variables
    metrics_port = int(os.getenv('PROMETHEUS_METRICS_PORT', '8010'))
    loki_url = os.getenv('LOKI_URL', 'http://localhost:3100')
    
    # Call the setup_monitoring function from the consolidated module
    return _setup_monitoring(
        service_name=service_name,
        enable_prometheus=enable_prometheus,
        enable_loki=enable_loki,
        metrics_port=metrics_port,
        loki_url=loki_url,
        default_labels=default_labels
    )

# Example usage
if __name__ == "__main__":
    # Set up monitoring for a service
    monitor, metrics = setup_monitoring(
        service_name="example-service",
        default_labels={"environment": "development"}
    )
    
    # Log some messages
    monitor.log_info("Service started", component="main")
    
    # Update some metrics
    monitor.increment_counter("requests_total", 1, method="GET", endpoint="/api/data", status="200")
    
    print("Monitoring initialized. Check Prometheus and Loki for metrics and logs.")
