#!/usr/bin/env python3
"""
Prometheus and Loki Utilities

This module provides utility functions for setting up Prometheus metrics
and Loki logging in various components of the NextGen FinGPT project.
"""

import os
import logging
import importlib.util
from typing import Optional, Dict, Any, Tuple

# Check if required modules are available
def _check_module(module_name: str) -> bool:
    """Check if a module is available."""
    return importlib.util.find_spec(module_name) is not None

# Define module availability flags
PROMETHEUS_AVAILABLE = _check_module("prometheus.prometheus_manager")
LOKI_AVAILABLE = _check_module("prometheus.loki.loki_manager")

# Import modules if available
if PROMETHEUS_AVAILABLE:
    from prometheus.prometheus_manager import PrometheusManager
if LOKI_AVAILABLE:
    from prometheus.loki.loki_manager import LokiManager

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prometheus_loki_utils")

class MonitoringManager:
    """
    A utility class for managing Prometheus metrics and Loki logging.
    
    This class provides a unified interface for setting up monitoring
    in various components of the NextGen FinGPT project.
    """
    
    def __init__(
        self, 
        service_name: str,
        enable_prometheus: bool = True,
        enable_loki: bool = True,
        metrics_port: Optional[int] = None,
        loki_url: Optional[str] = None,
        loki_username: Optional[str] = None,
        loki_password: Optional[str] = None,
        default_labels: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the monitoring manager.
        
        Args:
            service_name (str): Name of the service for metrics and logs identification.
            enable_prometheus (bool): Whether to enable Prometheus metrics. Defaults to True.
            enable_loki (bool): Whether to enable Loki logging. Defaults to True.
            metrics_port (int, optional): Port to expose Prometheus metrics on.
            loki_url (str, optional): URL of the Loki server.
            loki_username (str, optional): Username for Loki authentication.
            loki_password (str, optional): Password for Loki authentication.
            default_labels (dict, optional): Default labels to apply to all metrics and logs.
        """
        self.service_name = service_name
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_loki = enable_loki and LOKI_AVAILABLE
        self.default_labels = default_labels or {}
        
        # Initialize Prometheus manager if enabled
        self.prom = None
        if self.enable_prometheus:
            try:
                self.prom = PrometheusManager(
                    service_name=service_name,
                    expose_metrics=True,
                    metrics_port=metrics_port
                )
                logger.info(f"Prometheus metrics initialized for {service_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Prometheus metrics: {e}")
                self.enable_prometheus = False
        
        # Initialize Loki manager if enabled
        self.loki = None
        if self.enable_loki:
            try:
                self.loki = LokiManager(
                    service_name=service_name,
                    loki_url=loki_url,
                    loki_username=loki_username,
                    loki_password=loki_password
                )
                logger.info(f"Loki logging initialized for {service_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Loki logging: {e}")
                self.enable_loki = False
    
    def create_counter(self, name: str, description: str, labels: Optional[list] = None) -> Any:
        """
        Create a Prometheus counter metric.
        
        Args:
            name (str): Name of the counter.
            description (str): Description of what the counter measures.
            labels (list, optional): List of label names for this counter.
        
        Returns:
            Counter: A Prometheus Counter object, or None if Prometheus is not enabled.
        """
        if not self.enable_prometheus:
            return None
        
        return self.prom.create_counter(name, description, labels)
    
    def create_gauge(self, name: str, description: str, labels: Optional[list] = None) -> Any:
        """
        Create a Prometheus gauge metric.
        
        Args:
            name (str): Name of the gauge.
            description (str): Description of what the gauge measures.
            labels (list, optional): List of label names for this gauge.
        
        Returns:
            Gauge: A Prometheus Gauge object, or None if Prometheus is not enabled.
        """
        if not self.enable_prometheus:
            return None
        
        return self.prom.create_gauge(name, description, labels)
    
    def create_histogram(self, name: str, description: str, labels: Optional[list] = None, buckets: Optional[list] = None) -> Any:
        """
        Create a Prometheus histogram metric.
        
        Args:
            name (str): Name of the histogram.
            description (str): Description of what the histogram measures.
            labels (list, optional): List of label names for this histogram.
            buckets (list, optional): Custom buckets for the histogram.
        
        Returns:
            Histogram: A Prometheus Histogram object, or None if Prometheus is not enabled.
        """
        if not self.enable_prometheus:
            return None
        
        return self.prom.create_histogram(name, description, labels, buckets)
    
    def increment_counter(self, name: str, value: float = 1, **labels) -> bool:
        """
        Increment a counter by a given value.
        
        Args:
            name (str): Name of the counter to increment.
            value (float): Value to increment by. Defaults to 1.
            **labels: Labels to apply to this counter increment.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.enable_prometheus:
            return False
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        return self.prom.increment_counter(name, value, **all_labels)
    
    def set_gauge(self, name: str, value: float, **labels) -> bool:
        """
        Set a gauge to a given value.
        
        Args:
            name (str): Name of the gauge to set.
            value (float): Value to set.
            **labels: Labels to apply to this gauge.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.enable_prometheus:
            return False
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        return self.prom.set_gauge(name, value, **all_labels)
    
    def observe_histogram(self, name: str, value: float, **labels) -> bool:
        """
        Observe a value in a histogram.
        
        Args:
            name (str): Name of the histogram for observation.
            value (float): Value to observe.
            **labels: Labels to apply to this observation.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.enable_prometheus:
            return False
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        return self.prom.observe_histogram(name, value, **all_labels)
    
    def log_info(self, message: str, **labels) -> None:
        """
        Log an info message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Always log to standard logger
        logger.info(message)
        
        if not self.enable_loki:
            return
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        self.loki.info(message, **all_labels)
    
    def log_warning(self, message: str, **labels) -> None:
        """
        Log a warning message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Always log to standard logger
        logger.warning(message)
        
        if not self.enable_loki:
            return
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        self.loki.warning(message, **all_labels)
    
    def log_error(self, message: str, **labels) -> None:
        """
        Log an error message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Always log to standard logger
        logger.error(message)
        
        if not self.enable_loki:
            return
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        self.loki.error(message, **all_labels)
    
    def log_critical(self, message: str, **labels) -> None:
        """
        Log a critical message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Always log to standard logger
        logger.critical(message)
        
        if not self.enable_loki:
            return
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        self.loki.critical(message, **all_labels)
    
    def log_debug(self, message: str, **labels) -> None:
        """
        Log a debug message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Always log to standard logger
        logger.debug(message)
        
        if not self.enable_loki:
            return
        
        # Merge default labels with provided labels
        all_labels = {**self.default_labels, **labels}
        
        self.loki.debug(message, **all_labels)


# Convenience function to create a monitoring manager
def setup_monitoring(
    service_name: str,
    enable_prometheus: bool = True,
    enable_loki: bool = True,
    metrics_port: Optional[int] = None,
    loki_url: Optional[str] = None,
    default_labels: Optional[Dict[str, str]] = None
) -> Tuple[MonitoringManager, Dict[str, Any]]:
    """
    Set up monitoring for a service.
    
    This function creates a MonitoringManager instance and initializes
    common metrics for the service.
    
    Args:
        service_name (str): Name of the service for metrics and logs identification.
        enable_prometheus (bool): Whether to enable Prometheus metrics. Defaults to True.
        enable_loki (bool): Whether to enable Loki logging. Defaults to True.
        metrics_port (int, optional): Port to expose Prometheus metrics on.
        loki_url (str, optional): URL of the Loki server.
        default_labels (dict, optional): Default labels to apply to all metrics and logs.
    
    Returns:
        tuple: A tuple containing the MonitoringManager instance and a dictionary of metrics.
    """
    # Create the monitoring manager
    monitor = MonitoringManager(
        service_name=service_name,
        enable_prometheus=enable_prometheus,
        enable_loki=enable_loki,
        metrics_port=metrics_port,
        loki_url=loki_url,
        default_labels=default_labels
    )
    
    # Initialize common metrics
    metrics = {}
    
    if monitor.enable_prometheus:
        # Request counter
        metrics['requests_total'] = monitor.create_counter(
            "requests_total",
            "Total count of requests",
            ["method", "endpoint", "status"]
        )
        
        # Response time histogram
        metrics['response_time_seconds'] = monitor.create_histogram(
            "response_time_seconds",
            "Response time in seconds",
            ["method", "endpoint"]
        )
        
        # Error counter
        metrics['errors_total'] = monitor.create_counter(
            "errors_total",
            "Total count of errors",
            ["type", "code"]
        )
        
        # Active requests gauge
        metrics['active_requests'] = monitor.create_gauge(
            "active_requests",
            "Number of active requests",
            ["method", "endpoint"]
        )
    
    return monitor, metrics


# Example usage
if __name__ == "__main__":
    # Set up monitoring for a service
    monitor, metrics = setup_monitoring(
        service_name="example-service",
        metrics_port=9091,
        default_labels={"environment": "development"}
    )
    
    # Log some messages
    monitor.log_info("Service started", component="main")
    monitor.log_warning("Resource usage high", component="resource_monitor", usage=85)
    
    # Update some metrics
    monitor.increment_counter("requests_total", 1, method="GET", endpoint="/api/data", status="200")
    monitor.observe_histogram("response_time_seconds", 0.2, method="GET", endpoint="/api/data")
    
    print("Monitoring initialized. Check Prometheus and Loki for metrics and logs.")
