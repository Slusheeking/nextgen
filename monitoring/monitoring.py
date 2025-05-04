#!/usr/bin/env python3
"""
Unified Monitoring Module

This module provides a comprehensive monitoring solution for the NextGen FinGPT project,
combining Prometheus metrics and Loki logging functionality in a single package.
It includes:
- PrometheusManager: For collecting and exposing Prometheus metrics
- LokiManager: For sending logs to Loki
- MonitoringManager: A unified interface for both Prometheus and Loki
- SystemMetricsCollector: For collecting system metrics (CPU, memory, disk, GPU, etc.)
"""

import os
import time
import logging
import threading
import importlib.util
import platform
import subprocess
import socket
from urllib.parse import urlparse
from typing import Optional, Dict, Any, Tuple, List, Union
from threading import Thread

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    from prometheus_client import start_http_server, push_to_gateway
    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False

try:
    import logging_loki
    LOGGING_LOKI_AVAILABLE = True
except ImportError:
    LOGGING_LOKI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
    # Load environment variables if dotenv is available
    load_dotenv()
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        GPU_AVAILABLE = True
    except ImportError:
        GPU_AVAILABLE = False

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitoring")


class PrometheusManager:
    """
    A manager class for Prometheus metrics functionality.
    Provides an interface for creating and managing Prometheus metrics,
    as well as exposing them via HTTP server or pushing to a Pushgateway.
    """
    
    def __init__(self, service_name=None, expose_metrics=True, metrics_port=None, 
                 pushgateway_url=None, pushgateway_job=None):
        """
        Initialize the Prometheus metrics manager.
        
        Args:
            service_name (str, optional): Name of the service for metrics identification.
                                          Defaults to env var or 'nextgen'.
            expose_metrics (bool): Whether to expose metrics via HTTP server. Defaults to True.
            metrics_port (int, optional): Port to expose metrics on. Defaults to env var or 8010.
            pushgateway_url (str, optional): URL of Prometheus Pushgateway if using push model.
                                            Defaults to env var or None.
            pushgateway_job (str, optional): Job name for Pushgateway. Defaults to service_name.
        """
        if not PROMETHEUS_CLIENT_AVAILABLE:
            raise ImportError("prometheus_client library is not available. Please install it with: pip install prometheus_client")
        
        # Set service name from parameters or environment variables
        environment = os.getenv('ENVIRONMENT', 'development')
        service_version = os.getenv('SERVICE_VERSION', '1.0.0')
        default_name = f"nextgen-{environment}"
        self.service_name = service_name or os.getenv('SERVICE_NAME', default_name)
        
        # Set metrics configuration
        self.metrics_port = int(metrics_port or os.getenv('PROMETHEUS_METRICS_PORT', '8010'))
        self.pushgateway_url = pushgateway_url or os.getenv('PUSHGATEWAY_URL', None)
        self.pushgateway_job = pushgateway_job or self.service_name
        
        # Store environment info
        self.environment = environment
        self.version = service_version
        
        # Dictionary to store all metrics
        self.metrics = {}
        
        # Start metrics server if requested
        if expose_metrics:
            self._start_metrics_server()
        
        # Set up automatic push to gateway if URL provided
        self.push_thread = None
        if self.pushgateway_url:
            self.push_interval = int(os.getenv('PUSHGATEWAY_INTERVAL', '15'))
            self._start_push_thread()
    
    def _start_metrics_server(self):
        """Start the HTTP server to expose metrics to Prometheus."""
        start_http_server(self.metrics_port)
        logger.info(f"Prometheus metrics exposed on port {self.metrics_port}")
    
    def _start_push_thread(self):
        """Start background thread to periodically push metrics to Pushgateway."""
        if self.push_thread is not None and self.push_thread.is_alive():
            return  # Thread already running
        
        def push_metrics():
            while True:
                try:
                    push_to_gateway(
                        self.pushgateway_url, 
                        job=self.pushgateway_job,
                        registry=None  # Use the default registry
                    )
                except Exception as e:
                    logger.error(f"Error pushing to Pushgateway: {e}")
                
                time.sleep(self.push_interval)
        
        self.push_thread = threading.Thread(target=push_metrics, daemon=True)
        self.push_thread.start()
    
    def create_counter(self, name, description, labels=None):
        """
        Create a Prometheus Counter metric.
        
        Args:
            name (str): Name of the counter.
            description (str): Description of what the counter measures.
            labels (list, optional): List of label names for this counter.
        
        Returns:
            Counter: A Prometheus Counter object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        counter = Counter(metric_name, description, all_labels)
        self.metrics[name] = counter
        return counter
    
    def create_gauge(self, name, description, labels=None):
        """
        Create a Prometheus Gauge metric.
        
        Args:
            name (str): Name of the gauge.
            description (str): Description of what the gauge measures.
            labels (list, optional): List of label names for this gauge.
        
        Returns:
            Gauge: A Prometheus Gauge object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        gauge = Gauge(metric_name, description, all_labels)
        self.metrics[name] = gauge
        return gauge
    
    def create_histogram(self, name, description, labels=None, buckets=None):
        """
        Create a Prometheus Histogram metric.
        
        Args:
            name (str): Name of the histogram.
            description (str): Description of what the histogram measures.
            labels (list, optional): List of label names for this histogram.
            buckets (list, optional): Custom buckets for the histogram.
        
        Returns:
            Histogram: A Prometheus Histogram object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        kwargs = {}
        if buckets:
            kwargs['buckets'] = buckets
        
        histogram = Histogram(metric_name, description, all_labels, **kwargs)
        self.metrics[name] = histogram
        return histogram
    
    def create_summary(self, name, description, labels=None):
        """
        Create a Prometheus Summary metric.
        
        Args:
            name (str): Name of the summary.
            description (str): Description of what the summary measures.
            labels (list, optional): List of label names for this summary.
        
        Returns:
            Summary: A Prometheus Summary object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        summary = Summary(metric_name, description, all_labels)
        self.metrics[name] = summary
        return summary
    
    def get_metric(self, name):
        """
        Get a previously created metric by name.
        
        Args:
            name (str): Name of the metric to retrieve.
        
        Returns:
            The metric object if found, None otherwise.
        """
        return self.metrics.get(name)
    
    def increment_counter(self, name, value=1, **labels):
        """
        Increment a counter by a given value.
        
        Args:
            name (str): Name of the counter to increment.
            value (float): Value to increment by. Defaults to 1.
            **labels: Labels to apply to this counter increment.
        
        Returns:
            bool: True if successful, False if counter not found.
        """
        counter = self.get_metric(name)
        if not counter or not hasattr(counter, 'inc'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        counter.labels(**label_values).inc(value)
        return True
    
    def set_gauge(self, name, value, **labels):
        """
        Set a gauge to a given value.
        
        Args:
            name (str): Name of the gauge to set.
            value (float): Value to set.
            **labels: Labels to apply to this gauge.
        
        Returns:
            bool: True if successful, False if gauge not found.
        """
        gauge = self.get_metric(name)
        if not gauge or not hasattr(gauge, 'set'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        gauge.labels(**label_values).set(value)
        return True
    
    def observe_histogram(self, name, value, **labels):
        """
        Observe a value in a histogram.
        
        Args:
            name (str): Name of the histogram for observation.
            value (float): Value to observe.
            **labels: Labels to apply to this observation.
        
        Returns:
            bool: True if successful, False if histogram not found.
        """
        histogram = self.get_metric(name)
        if not histogram or not hasattr(histogram, 'observe'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        histogram.labels(**label_values).observe(value)
        return True
    
    def observe_summary(self, name, value, **labels):
        """
        Observe a value in a summary.
        
        Args:
            name (str): Name of the summary for observation.
            value (float): Value to observe.
            **labels: Labels to apply to this observation.
        
        Returns:
            bool: True if successful, False if summary not found.
        """
        summary = self.get_metric(name)
        if not summary or not hasattr(summary, 'observe'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        summary.labels(**label_values).observe(value)
        return True
    
    def force_push(self):
        """
        Force a push to the Pushgateway immediately.
        
        Returns:
            bool: True if successful, False if no Pushgateway URL configured.
        """
        if not self.pushgateway_url:
            return False
        
        try:
            push_to_gateway(
                self.pushgateway_url, 
                job=self.pushgateway_job,
                registry=None  # Use the default registry
            )
            return True
        except Exception as e:
            logger.error(f"Error pushing to Pushgateway: {e}")
            return False


class LokiManager:
    """
    A manager class for Loki logging functionality.
    Implements a wrapper around python-logging-loki to send logs to a Loki instance.
    """
    
    def __init__(self, service_name=None, loki_url=None, loki_username=None, loki_password=None):
        """
        Initialize the Loki logger manager.
        
        Args:
            service_name (str, optional): Name of the service for log identification.
            loki_url (str, optional): URL of the Loki server including protocol, host and port.
                                     Defaults to env var LOKI_URL or 'http://localhost:3100'.
            loki_username (str, optional): Username for Loki authentication. Defaults to env var.
            loki_password (str, optional): Password for Loki authentication. Defaults to env var.
        """
        if not LOGGING_LOKI_AVAILABLE:
            raise ImportError("logging_loki library is not available. Please install it with: pip install python-logging-loki")
        
        # Set service name from parameters or environment variables
        environment = os.getenv('ENVIRONMENT', 'development')
        service_version = os.getenv('SERVICE_VERSION', '1.0.0')
        default_name = f"nextgen-{environment}"
        self.service_name = service_name or os.getenv('SERVICE_NAME', default_name)
        
        # Parse Loki URL from parameters or environment variables
        self.loki_url = loki_url or os.getenv('LOKI_URL', 'http://localhost:3100')
        
        # Parse the URL to extract host and port
        parsed_url = urlparse(self.loki_url)
        self.loki_host = parsed_url.hostname or 'localhost'
        self.loki_port = parsed_url.port or 3100
        self.loki_scheme = parsed_url.scheme or 'http'
        
        # Set authentication credentials
        self.loki_username = loki_username or os.getenv('LOKI_USERNAME', '')
        self.loki_password = loki_password or os.getenv('LOKI_PASSWORD', '')
        
        # Add version as a default tag
        self.version = service_version
        
        # Configure the Loki handler
        self._configure_loki_handler()
        
    def _configure_loki_handler(self):
        """Configure the Loki handler with current settings."""
        # Create authentication if credentials provided
        auth = None
        if self.loki_username and self.loki_password:
            auth = (self.loki_username, self.loki_password)
        
        # Create the full URL
        push_url = f"{self.loki_scheme}://{self.loki_host}:{self.loki_port}/loki/api/v1/push"
        
        # Create the Loki handler
        loki_handler = logging_loki.LokiHandler(
            url=push_url,
            tags={
                "service": self.service_name,
                "version": self.version,
                "environment": os.getenv('ENVIRONMENT', 'development')
            },
            auth=auth,
            version="1"
        )
        
        # Configure the root logger with the Loki handler
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(loki_handler)
        
        # Add a formatter for structured logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        loki_handler.setFormatter(formatter)
    
    def info(self, message, **labels):
        """
        Log an info message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.INFO, message, **labels)
    
    def warning(self, message, **labels):
        """
        Log a warning message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.WARNING, message, **labels)
    
    def error(self, message, **labels):
        """
        Log an error message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.ERROR, message, **labels)
    
    def critical(self, message, **labels):
        """
        Log a critical message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.CRITICAL, message, **labels)
    
    def debug(self, message, **labels):
        """
        Log a debug message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.DEBUG, message, **labels)
    
    def _log(self, level, message, **labels):
        """
        Internal method to log messages with additional labels.
        
        Args:
            level (int): Logging level (e.g., logging.INFO).
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Add timestamp and service labels
        extra_labels = {
            'timestamp': time.time(),
            'service': self.service_name
        }
        
        # Add any additional labels provided
        extra_labels.update(labels)
        
        # Log the message with the extra labels
        self.logger.log(level, message, extra=extra_labels)
    
    def reconfigure(self, loki_url=None, loki_username=None, 
                   loki_password=None, service_name=None):
        """
        Reconfigure the Loki logger with new settings.
        
        Args:
            loki_url (str, optional): New URL of the Loki server including protocol, host and port.
            loki_username (str, optional): New username for Loki authentication.
            loki_password (str, optional): New password for Loki authentication.
            service_name (str, optional): New service name.
        """
        if loki_url:
            self.loki_url = loki_url
            # Re-parse the URL
            parsed_url = urlparse(self.loki_url)
            self.loki_host = parsed_url.hostname or 'localhost'
            self.loki_port = parsed_url.port or 3100
            self.loki_scheme = parsed_url.scheme or 'http'
            
        if loki_username:
            self.loki_username = loki_username
        if loki_password:
            self.loki_password = loki_password
        if service_name:
            self.service_name = service_name
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Reconfigure with new settings
        self._configure_loki_handler()


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
        self.enable_prometheus = enable_prometheus and PROMETHEUS_CLIENT_AVAILABLE
        self.enable_loki = enable_loki and LOGGING_LOKI_AVAILABLE
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


class SystemMetricsCollector:
    """
    Collects system metrics and exposes them through Prometheus.
    Sends system alerts and notifications to Loki.
    """
    
    def __init__(self, interval=5, service_name="system_metrics", enable_loki=True):
        """
        Initialize the system metrics collector.
        
        Args:
            interval (int): Collection interval in seconds. Defaults to 5.
            service_name (str): Name of the service for Prometheus. Defaults to "system_metrics".
            enable_loki (bool): Whether to enable Loki logging for alerts. Defaults to True.
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil library is not available. Please install it with: pip install psutil")
            
        self.interval = interval
        self.running = False
        self.metrics_thread = None
        
        # Initialize Prometheus manager
        self.prom = PrometheusManager(service_name=service_name)
        
        # Initialize Loki manager for alerts
        self.enable_loki = enable_loki
        if enable_loki:
            self.loki = LokiManager(service_name=service_name)
            logger.info("Loki logging initialized for system alerts")
        
        # Initialize metrics
        self._init_metrics()
        
        # Configure alert thresholds with default values
        self._configure_alert_thresholds()
        
        # Initialize GPU monitoring if available
        self.has_gpu = False
        if GPU_AVAILABLE or NVML_AVAILABLE:
            try:
                if NVML_AVAILABLE:
                    nvml.nvmlInit()
                    self.gpu_count = nvml.nvmlDeviceGetCount()
                    self.has_gpu = self.gpu_count > 0
                    if self.has_gpu:
                        logger.info(f"NVML initialized. Found {self.gpu_count} GPU(s)")
                elif GPU_AVAILABLE:
                    if 'GPUtil' in globals():
                        self.gpus = GPUtil.getGPUs()
                        self.has_gpu = len(self.gpus) > 0
                        if self.has_gpu:
                            logger.info(f"GPUtil initialized. Found {len(self.gpus)} GPU(s)")
                    else:  # pynvml
                        import pynvml
                        pynvml.nvmlInit()
                        self.gpu_count = pynvml.nvmlDeviceGetCount()
                        self.has_gpu = self.gpu_count > 0
                        if self.has_gpu:
                            logger.info(f"PYNVML initialized. Found {self.gpu_count} GPU(s)")
            except Exception as e:
                logger.error(f"Error initializing GPU monitoring: {e}")
                self.has_gpu = False
        
        if not self.has_gpu:
            logger.warning("No GPU detected or GPU monitoring libraries unavailable")
    
    def _configure_alert_thresholds(self):
        """Configure default alert thresholds for system metrics."""
        self.alert_thresholds = {
            # CPU related thresholds
            'cpu_utilization_percent': {
                'warning': 80.0,  # 80% utilization
                'critical': 95.0,  # 95% utilization
                'duration': 3,     # Alert after 3 consecutive readings
                'counter': 0,
                'last_alert': 0,   # Timestamp of last alert
                'cooldown': 300    # 5 minutes between alerts
            },
            # Memory related thresholds
            'memory_usage_percent': {
                'warning': 80.0,
                'critical': 95.0,
                'duration': 3,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 300
            },
            # Disk related thresholds
            'disk_usage_percent': {
                'warning': 80.0,
                'critical': 95.0,
                'duration': 1,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 3600  # 1 hour between disk alerts
            },
            # GPU related thresholds
            'gpu_utilization_percent': {
                'warning': 85.0,
                'critical': 98.0,
                'duration': 5,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 300
            },
            'gpu_memory_percent': {
                'warning': 80.0,
                'critical': 95.0,
                'duration': 3,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 300
            },
            'gpu_temperature_celsius': {
                'warning': 80.0,    # 80°C is getting hot
                'critical': 90.0,   # 90°C is too hot
                'duration': 1,      # Alert immediately on high temperature
                'counter': 0,
                'last_alert': 0,
                'cooldown': 60      # 1 minute between temperature alerts
            }
        }
        
        logger.info("Alert thresholds configured with default values")
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        # CPU metrics
        self.cpu_percent = self.prom.create_gauge(
            "cpu_utilization_percent", 
            "CPU utilization percentage",
            ["cpu"]
        )
        
        self.cpu_freq = self.prom.create_gauge(
            "cpu_frequency_mhz",
            "CPU frequency in MHz",
            ["cpu"]
        )
        
        self.cpu_count = self.prom.create_gauge(
            "cpu_count",
            "Number of CPU cores/threads"
        )
        
        self.load_avg = self.prom.create_gauge(
            "load_average",
            "System load average",
            ["period"]
        )
        
        self.cpu_temp = self.prom.create_gauge(
            "cpu_temperature_celsius",
            "CPU temperature in Celsius",
            ["sensor"]
        )
        
        # Memory metrics
        self.memory_usage = self.prom.create_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["type"]
        )
        
        self.memory_percent = self.prom.create_gauge(
            "memory_usage_percent",
            "Memory usage percentage",
            ["type"]
        )
        
        # Disk metrics
        self.disk_usage = self.prom.create_gauge(
            "disk_usage_bytes",
            "Disk usage in bytes",
            ["device", "mountpoint", "type"]
        )
        
        self.disk_percent = self.prom.create_gauge(
            "disk_usage_percent",
            "Disk usage percentage",
            ["device", "mountpoint"]
        )
        
        self.disk_io = self.prom.create_gauge(
            "disk_io_bytes",
            "Disk I/O in bytes",
            ["device", "direction"]
        )
        
        # Network metrics
        self.network_io = self.prom.create_gauge(
            "network_io_bytes",
            "Network I/O in bytes",
            ["interface", "direction"]
        )
        
        self.network_connections = self.prom.create_gauge(
            "network_connections",
            "Number of network connections",
            ["type", "status"]
        )
        
        # GPU metrics (if available)
        if GPU_AVAILABLE or NVML_AVAILABLE:
            self.gpu_utilization = self.prom.create_gauge(
                "gpu_utilization_percent",
                "GPU utilization percentage",
                ["gpu", "type"]
            )
            
            self.gpu_memory = self.prom.create_gauge(
                "gpu_memory_bytes",
                "GPU memory usage in bytes",
                ["gpu", "type"]
            )
            
            self.gpu_memory_percent = self.prom.create_gauge(
                "gpu_memory_percent",
                "GPU memory usage percentage",
                ["gpu"]
            )
            
            self.gpu_temp = self.prom.create_gauge(
                "gpu_temperature_celsius",
                "GPU temperature in Celsius",
                ["gpu"]
            )
            
            self.gpu_power = self.prom.create_gauge(
                "gpu_power_watts",
                "GPU power usage in watts",
                ["gpu", "type"]
            )
        
        # System uptime
        self.uptime = self.prom.create_gauge(
            "system_uptime_seconds",
            "System uptime in seconds"
        )
        
        # Process count
        self.process_count = self.prom.create_gauge(
            "process_count",
            "Number of running processes",
            ["state"]
        )
    
    def check_and_alert(self, metric_name, value, labels=None):
        """
        Check if a metric value exceeds alert thresholds and send alert if needed.
        
        Args:
            metric_name (str): Name of the metric
            value (float): Current value of the metric
            labels (dict, optional): Additional labels for this metric
        """
        if not self.enable_loki or metric_name not in self.alert_thresholds:
            return
            
        thresholds = self.alert_thresholds[metric_name]
        current_time = time.time()
        
        # Format labels for alert message
        label_str = ""
        if labels:
            label_str = " (" + ", ".join(f"{k}={v}" for k, v in labels.items()) + ")"
        
        # Check if value exceeds thresholds
        if value >= thresholds['critical']:
            thresholds['counter'] += 1
            
            if (thresholds['counter'] >= thresholds['duration'] and 
                current_time - thresholds['last_alert'] > thresholds['cooldown']):
                # Send CRITICAL alert to Loki
                alert_msg = f"CRITICAL: {metric_name}{label_str} = {value:.2f} exceeds critical threshold of {thresholds['critical']}"
                self.loki.critical(alert_msg, metric=metric_name, value=value, threshold="critical", **labels if labels else {})
                logger.critical(alert_msg)
                
                # Reset counter and update last alert time
                thresholds['counter'] = 0
                thresholds['last_alert'] = current_time
                
        elif value >= thresholds['warning']:
            thresholds['counter'] += 1
            
            if (thresholds['counter'] >= thresholds['duration'] and 
                current_time - thresholds['last_alert'] > thresholds['cooldown']):
                # Send WARNING alert to Loki
                alert_msg = f"WARNING: {metric_name}{label_str} = {value:.2f} exceeds warning threshold of {thresholds['warning']}"
                self.loki.warning(alert_msg, metric=metric_name, value=value, threshold="warning", **labels if labels else {})
                logger.warning(alert_msg)
                
                # Reset counter and update last alert time
                thresholds['counter'] = 0
                thresholds['last_alert'] = current_time
                
        else:
            # Reset counter if value returns to normal
            if thresholds['counter'] > 0:
                thresholds['counter'] = 0
                
                # Log recovery if we were previously above thresholds
                if current_time - thresholds['last_alert'] < thresholds['cooldown']:
                    recovery_msg = f"RECOVERY: {metric_name}{label_str} = {value:.2f} returned to normal"
                    if self.enable_loki:
                        self.loki.info(recovery_msg, metric=metric_name, value=value, status="recovery", **labels if labels else {})
                    logger.info(recovery_msg)


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
