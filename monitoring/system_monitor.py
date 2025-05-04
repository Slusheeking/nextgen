"""
Simplified System Monitor for NextGen

Provides:
- Unified MonitoringManager for Prometheus metrics and Loki logging
- Basic SystemMetricsCollector for CPU, memory, and disk metrics

Dependencies:
- prometheus_client
- psutil

Usage:
    from monitoring.system_monitor import MonitoringManager, SystemMetricsCollector

    monitor = MonitoringManager(service_name="my-service")
    monitor.log_info("Service started")
    monitor.increment_counter("requests_total", method="GET", endpoint="/api/data", status="200")

    # Start system metrics collection (optional)
    collector = SystemMetricsCollector(monitor)
    collector.start()
    
This module is designed to be a simplified, consolidated replacement for the previous
monitoring setup that was spread across multiple files. It provides a clean, easy-to-use
interface for both metrics and logging.
"""

import time
import logging
import threading
import os
import socket
import traceback
import psutil
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server


class MonitoringManager:
    """
    Unified monitoring manager that handles both metrics (Prometheus) and logging (Python logging).
    
    This class provides a simplified interface for monitoring and observability,
    combining metrics collection and structured logging in one place.
    """
    def __init__(self, service_name, metrics_port=8010):
        """
        Initialize the monitoring manager.

        Args:
            service_name: Name of the service being monitored
            metrics_port: Port for Prometheus metrics server
        """
        # Sanitize service name for Prometheus metrics (no hyphens)
        self.service_name = service_name
        self.metrics_name = service_name.replace('-', '_')
        self.metrics_port = metrics_port
        self.hostname = socket.gethostname()
        self.metrics = {}
        self.start_time = datetime.now()

        # Set up basic logging first
        self._setup_basic_logging()

        # Prometheus setup - try to start the server, but continue if it fails
        try:
            # Try to start the metrics server, but don't fail if the port is already in use
            try:
                start_http_server(self.metrics_port)
                self.log_info(
                    f"Prometheus metrics server started on port {self.metrics_port}",
                    component="monitoring",
                    action="metrics_server_start"
                )
            except OSError as e:
                if "Address already in use" in str(e):
                    self.log_warning(
                        f"Prometheus metrics port {self.metrics_port} already in use. Using existing server.",
                        component="monitoring",
                        action="metrics_server_warning"
                    )
                else:
                    raise e
            
            # Initialize metrics
            self._init_metrics()
        except Exception as e:
            self.log_error(
                f"Failed to initialize Prometheus metrics: {e}",
                component="monitoring",
                action="metrics_init_error",
                error=str(e)
            )

        # File logging setup
        self._setup_file_logging()
        
        # Log initialization
        self.log_info(
            f"MonitoringManager initialized for service: {service_name}",
            component="monitoring",
            action="initialization"
        )
        
    def _setup_basic_logging(self):
        """Set up basic console logging."""
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.INFO)
        
        # Check if handler already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_file_logging(self):
        """Set up rotating file logging in addition to console logging."""
        import logging
        import os
        from logging.handlers import RotatingFileHandler

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up service-specific log file
        log_path = os.path.join(log_dir, f"{self.service_name}.log")
        file_handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Set up master log file
        master_log_path = os.path.join(log_dir, "master.log")
        master_handler = RotatingFileHandler(master_log_path, maxBytes=10*1024*1024, backupCount=10)
        master_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        master_handler.setFormatter(master_formatter)
        self.logger.addHandler(master_handler)
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # Request metrics
            self.metrics["requests_total"] = Counter(
                f"{self.metrics_name}_requests_total",
                "Total count of requests",
                ["method", "endpoint", "status"]
            )
            self.metrics["request_duration_seconds"] = Histogram(
                f"{self.metrics_name}_request_duration_seconds",
                "Request duration in seconds",
                ["method", "endpoint"]
            )
            self.metrics["active_requests"] = Gauge(
                f"{self.metrics_name}_active_requests",
                "Number of active requests",
                ["method", "endpoint"]
            )
            
            # Error metrics
            self.metrics["errors_total"] = Counter(
                f"{self.metrics_name}_errors_total",
                "Total count of errors",
                ["component", "type"]
            )
            
            # System metrics
            self.metrics["cpu_percent"] = Gauge(
                f"{self.metrics_name}_cpu_percent",
                "CPU utilization percent"
            )
            self.metrics["memory_percent"] = Gauge(
                f"{self.metrics_name}_memory_percent",
                "Memory utilization percent"
            )
            self.metrics["disk_percent"] = Gauge(
                f"{self.metrics_name}_disk_percent",
                "Disk utilization percent"
            )
            
            # Application metrics
            self.metrics["uptime_seconds"] = Gauge(
                f"{self.metrics_name}_uptime_seconds",
                "Service uptime in seconds"
            )
            self.metrics["operations_total"] = Counter(
                f"{self.metrics_name}_operations_total",
                "Total count of operations",
                ["component", "operation"]
            )
            self.metrics["operation_duration_seconds"] = Summary(
                f"{self.metrics_name}_operation_duration_seconds",
                "Operation duration in seconds",
                ["component", "operation"]
            )
        except Exception as e:
            # If we can't set up metrics, log the error but continue
            if hasattr(self, 'logger'):
                self.log_error(
                    f"Failed to initialize metrics: {e}",
                    component="monitoring",
                    action="metrics_init_error",
                    error=str(e)
                )

    # Logging methods
    def log_info(self, message, **labels):
        """Log an info message with additional labels."""
        try:
            self.logger.info(message, extra={"tags": labels})
            # Also increment operation counter if component and action are provided
            if "component" in labels and "action" in labels:
                self.increment_counter(
                    "operations_total", 
                    component=labels["component"], 
                    operation=labels["action"]
                )
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.info(f"{message} (labels: {labels}) [Logging error: {e}]")

    def log_warning(self, message, **labels):
        """Log a warning message with additional labels."""
        try:
            self.logger.warning(message, extra={"tags": labels})
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.warning(f"{message} (labels: {labels}) [Logging error: {e}]")

    def log_error(self, message, **labels):
        """Log an error message with additional labels."""
        try:
            # Add stack trace if not provided
            if "stack_trace" not in labels:
                labels["stack_trace"] = traceback.format_exc()
                
            self.logger.error(message, extra={"tags": labels})
            
            # Also increment error counter if component is provided
            if "component" in labels:
                error_type = labels.get("error_type", "general")
                self.increment_counter("errors_total", component=labels["component"], type=error_type)
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.error(f"{message} (labels: {labels}) [Logging error: {e}]")

    def log_critical(self, message, **labels):
        """Log a critical message with additional labels."""
        try:
            # Add stack trace if not provided
            if "stack_trace" not in labels:
                labels["stack_trace"] = traceback.format_exc()
                
            self.logger.critical(message, extra={"tags": labels})
            
            # Also increment error counter if component is provided
            if "component" in labels:
                error_type = labels.get("error_type", "critical")
                self.increment_counter("errors_total", component=labels["component"], type=error_type)
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.critical(f"{message} (labels: {labels}) [Logging error: {e}]")

    def log_debug(self, message, **labels):
        """Log a debug message with additional labels."""
        try:
            self.logger.debug(message, extra={"tags": labels})
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.debug(f"{message} (labels: {labels}) [Logging error: {e}]")

    # Metric methods
    def increment_counter(self, name, value=1, **labels):
        """Increment a counter metric."""
        try:
            if name in self.metrics and isinstance(self.metrics[name], Counter):
                self.metrics[name].labels(**labels).inc(value)
        except Exception as e:
            self.log_error(
                f"Failed to increment counter {name}: {e}",
                component="monitoring",
                action="increment_counter_error",
                metric_name=name,
                labels=str(labels),
                error=str(e)
            )

    def set_gauge(self, name, value, **labels):
        """Set a gauge metric value."""
        try:
            if name in self.metrics and isinstance(self.metrics[name], Gauge):
                if labels:
                    self.metrics[name].labels(**labels).set(value)
                else:
                    self.metrics[name].set(value)
        except Exception as e:
            self.log_error(
                f"Failed to set gauge {name}: {e}",
                component="monitoring",
                action="set_gauge_error",
                metric_name=name,
                labels=str(labels),
                error=str(e)
            )
            
    def observe_histogram(self, name, value, **labels):
        """Observe a value for a histogram metric."""
        try:
            if name in self.metrics and isinstance(self.metrics[name], Histogram):
                self.metrics[name].labels(**labels).observe(value)
        except Exception as e:
            self.log_error(
                f"Failed to observe histogram {name}: {e}",
                component="monitoring",
                action="observe_histogram_error",
                metric_name=name,
                labels=str(labels),
                error=str(e)
            )
            
    def observe_summary(self, name, value, **labels):
        """Observe a value for a summary metric."""
        try:
            if name in self.metrics and isinstance(self.metrics[name], Summary):
                self.metrics[name].labels(**labels).observe(value)
        except Exception as e:
            self.log_error(
                f"Failed to observe summary {name}: {e}",
                component="monitoring",
                action="observe_summary_error",
                metric_name=name,
                labels=str(labels),
                error=str(e)
            )
    
    def update_uptime(self):
        """Update the uptime metric."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            self.set_gauge("uptime_seconds", uptime)
        except Exception as e:
            self.log_error(
                f"Failed to update uptime: {e}",
                component="monitoring",
                action="update_uptime_error",
                error=str(e)
            )

class SystemMetricsCollector:
    """
    Collects system metrics like CPU, memory, and disk usage.
    
    This class runs in a background thread and periodically updates
    system metrics in the monitoring manager.
    """
    
    def __init__(self, monitor: MonitoringManager, interval=5):
        """
        Initialize the system metrics collector.
        
        Args:
            monitor: The monitoring manager to update metrics in
            interval: Collection interval in seconds
        """
        self.monitor = monitor
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """Start the metrics collection thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_loop, daemon=True)
            self.thread.start()
            self.monitor.log_info(
                f"System metrics collector started with interval {self.interval}s",
                component="system_metrics",
                action="collector_start",
                interval=str(self.interval)
            )

    def stop(self):
        """Stop the metrics collection thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            self.monitor.log_info(
                "System metrics collector stopped",
                component="system_metrics",
                action="collector_stop"
            )

    def _collect_loop(self):
        """Main collection loop that runs in a background thread."""
        while self.running:
            try:
                # Update uptime
                self.monitor.update_uptime()
                
                # CPU
                cpu = psutil.cpu_percent()
                self.monitor.set_gauge("cpu_percent", cpu)
                
                # Memory
                mem = psutil.virtual_memory().percent
                self.monitor.set_gauge("memory_percent", mem)
                
                # Disk
                disk = psutil.disk_usage('/').percent
                self.monitor.set_gauge("disk_percent", disk)
                
                # Sleep until next collection
                time.sleep(self.interval)
                
            except Exception as e:
                self.monitor.log_error(
                    f"Error collecting system metrics: {e}",
                    component="system_metrics",
                    action="collection_error",
                    error=str(e)
                )
                # Still sleep to avoid tight loop if there's a persistent error
                time.sleep(self.interval)
