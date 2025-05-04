"""
NextGen FinGPT Monitoring Module

This module provides comprehensive monitoring capabilities for the NextGen FinGPT project,
including Prometheus metrics collection, Loki logging, and system metrics monitoring.
It exposes a clean API for frontend integration and dashboard connectivity.
"""

# Export the main components from the system_monitor module
from monitoring.system_monitor import (
    MonitoringManager,
    SystemMetricsCollector
)

# Export the setup_monitoring function for backward compatibility
def setup_monitoring(service_name, enable_prometheus=True, enable_loki=True, default_labels=None):
    """
    Set up monitoring for a service (backward compatibility function).
    
    Args:
        service_name: Name of the service being monitored
        enable_prometheus: Whether to enable Prometheus metrics
        enable_loki: Whether to enable Loki logging
        default_labels: Default labels to apply to all metrics and logs
        
    Returns:
        Tuple of (MonitoringManager, metrics_dict)
    """
    monitor = MonitoringManager(
        service_name=service_name,
        loki_labels=default_labels
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
__version__ = '2.0.0'

# Define what gets imported with "from monitoring import *"
__all__ = [
    'MonitoringManager',
    'SystemMetricsCollector',
    'setup_monitoring',
]
