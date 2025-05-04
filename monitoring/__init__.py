"""
NextGen FinGPT Monitoring Module

This module provides comprehensive monitoring capabilities for the NextGen FinGPT project,
including Prometheus metrics collection, Loki logging, and system metrics monitoring.
It exposes a clean API for frontend integration and dashboard connectivity.
"""

# Export the main components from the consolidated monitoring module
from monitoring.monitoring import (
    PrometheusManager,
    LokiManager,
    MonitoringManager,
    SystemMetricsCollector,
    setup_monitoring
)

# Export the compatibility layer for existing code
from monitoring.prometheus_loki_utils import setup_monitoring as setup_prometheus_loki

# Export the system metrics collector for standalone usage
from monitoring.system_metrics_collector import SystemMetricsService

# Define the version
__version__ = '1.0.0'

# Define what gets imported with "from monitoring import *"
__all__ = [
    'PrometheusManager',
    'LokiManager',
    'MonitoringManager',
    'SystemMetricsCollector',
    'SystemMetricsService',
    'setup_monitoring',
    'setup_prometheus_loki',
]
