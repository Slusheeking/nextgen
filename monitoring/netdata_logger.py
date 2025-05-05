"""
Netdata Logger - Unified interface for Netdata metrics and Python logging

This module provides a unified interface for both Netdata metrics collection
and Python logging. It allows you to:

1. Send custom metrics to Netdata via StatsD protocol
2. Log messages with structured data using Python's logging system
3. Monitor system resources automatically

Usage:
    from monitoring.netdata_logger import NetdataLogger

    # Create a logger for your component
    logger = NetdataLogger(component_name="my-component")

    # Log messages
    logger.info("Service started")
    logger.warning("Resource usage high", cpu=85.2, memory=75.8)
    logger.error("Database connection failed", db_host="db.example.com")

    # Send metrics to Netdata
    logger.gauge("api_response_time", 0.235)
    logger.counter("api_requests", 1)
    logger.histogram("query_time", 0.056)

    # Start system metrics collection
    logger.start_system_metrics()
"""

import logging
import socket
import os
import time
import threading
import traceback
import json
import psutil
from datetime import datetime
from logging.handlers import RotatingFileHandler


class NetdataLogger:
    """
    Unified interface for Netdata metrics and Python logging
    """
    
    def __init__(self, component_name, log_dir=None, statsd_host="localhost", statsd_port=8125):
        """
        Initialize the Netdata Logger
        
        Args:
            component_name: Name of the component (used for both logging and metrics)
            log_dir: Directory for log files (default: PROJECT_ROOT/logs)
            statsd_host: Host for StatsD metrics (default: localhost)
            statsd_port: Port for StatsD metrics (default: 8125)
        """
        self.component_name = component_name
        self.statsd_host = statsd_host
        self.statsd_port = statsd_port
        self.metrics_prefix = component_name.replace('-', '_').replace('.', '_')
        self.start_time = datetime.now()
        self.system_metrics_thread = None
        self.system_metrics_running = False
        
        # Set up logging
        self.logger = logging.getLogger(component_name)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if no handlers exist
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Set up file logging if log_dir is provided or use default
        if log_dir is None:
            # Get project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            log_dir = os.path.join(project_root, "logs")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Component-specific log file
        log_file = os.path.join(log_dir, f"{component_name}.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Master log file
        master_log = os.path.join(log_dir, "master.log")
        master_handler = RotatingFileHandler(master_log, maxBytes=10*1024*1024, backupCount=10)
        master_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        master_handler.setFormatter(master_formatter)
        self.logger.addHandler(master_handler)
        
        self.info(f"NetdataLogger initialized for component: {component_name}")
    
    # Logging methods
    
    def _log(self, level, message, **kwargs):
        """
        Log a message with the specified level and additional data
        
        Args:
            level: Log level (e.g., "info", "warning", "error")
            message: Log message
            **kwargs: Additional data to include in the log
        """
        try:
            # Convert kwargs to a tags dictionary for structured logging
            self.logger.log(
                getattr(logging, level.upper()),
                message,
                extra={"tags": kwargs}
            )
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.log(
                getattr(logging, level.upper()),
                f"{message} (data: {kwargs}) [Logging error: {e}]"
            )
    
    def debug(self, message, **kwargs):
        """Log a debug message with additional data"""
        self._log("debug", message, **kwargs)
    
    def info(self, message, **kwargs):
        """Log an info message with additional data"""
        self._log("info", message, **kwargs)
    
    def warning(self, message, **kwargs):
        """Log a warning message with additional data"""
        self._log("warning", message, **kwargs)
    
    def error(self, message, **kwargs):
        """Log an error message with additional data"""
        if "stack_trace" not in kwargs:
            kwargs["stack_trace"] = traceback.format_exc()
        self._log("error", message, **kwargs)
    
    def critical(self, message, **kwargs):
        """Log a critical message with additional data"""
        if "stack_trace" not in kwargs:
            kwargs["stack_trace"] = traceback.format_exc()
        self._log("critical", message, **kwargs)
    
    # Netdata metrics methods
    
    def _send_statsd(self, metric_name, value, metric_type):
        """
        Send a metric to Netdata via StatsD protocol
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
            metric_type: Type of the metric (g=gauge, c=counter, ms=timing, h=histogram)
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message = f"{self.metrics_prefix}.{metric_name}:{value}|{metric_type}"
            sock.sendto(message.encode(), (self.statsd_host, self.statsd_port))
            return True
        except Exception as e:
            self.error(f"Failed to send metric to Netdata: {metric_name}", 
                      error=str(e), metric_value=value, metric_type=metric_type)
            return False
    
    def gauge(self, name, value):
        """
        Send a gauge metric to Netdata
        
        Gauges represent a value that can go up and down
        """
        return self._send_statsd(name, value, "g")
    
    def counter(self, name, value=1):
        """
        Send a counter metric to Netdata
        
        Counters represent values that only increase
        """
        return self._send_statsd(name, value, "c")
    
    def timing(self, name, value):
        """
        Send a timing metric to Netdata
        
        Timing metrics represent durations in milliseconds
        """
        return self._send_statsd(name, value, "ms")
    
    def histogram(self, name, value):
        """
        Send a histogram metric to Netdata
        
        Histograms track the distribution of values
        """
        return self._send_statsd(name, value, "h")
    
    # System metrics collection
    
    def start_system_metrics(self, interval=5):
        """
        Start collecting system metrics
        
        Args:
            interval: Collection interval in seconds (default: 5)
        """
        if self.system_metrics_thread is not None and self.system_metrics_thread.is_alive():
            self.info("System metrics collection is already running")
            return
        
        self.system_metrics_running = True
        self.system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            args=(interval,),
            daemon=True
        )
        self.system_metrics_thread.start()
        self.info("System metrics collection started", interval=interval)
    
    def stop_system_metrics(self):
        """Stop collecting system metrics"""
        if self.system_metrics_thread is None or not self.system_metrics_thread.is_alive():
            self.info("System metrics collection is not running")
            return
        
        self.system_metrics_running = False
        self.system_metrics_thread.join()
        self.info("System metrics collection stopped")
    
    def _collect_system_metrics(self, interval):
        """
        Collect system metrics at regular intervals
        
        Args:
            interval: Collection interval in seconds
        """
        while self.system_metrics_running:
            try:
                # Calculate uptime
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.gauge("uptime_seconds", uptime)
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent()
                self.gauge("cpu_percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.gauge("memory_percent", memory.percent)
                self.gauge("memory_used_mb", memory.used / (1024 * 1024))
                self.gauge("memory_available_mb", memory.available / (1024 * 1024))
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.gauge("disk_percent", disk.percent)
                self.gauge("disk_used_gb", disk.used / (1024 * 1024 * 1024))
                self.gauge("disk_free_gb", disk.free / (1024 * 1024 * 1024))
                
                # Network metrics
                net_io = psutil.net_io_counters()
                self.gauge("net_bytes_sent", net_io.bytes_sent)
                self.gauge("net_bytes_recv", net_io.bytes_recv)
                
                # Log summary of metrics
                self.debug(
                    "System metrics collected",
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_percent=disk.percent
                )
                
                # Sleep until next collection
                time.sleep(interval)
                
            except Exception as e:
                self.error(f"Error collecting system metrics: {e}")
                time.sleep(interval)  # Sleep even on error to avoid tight loop


# Example usage
if __name__ == "__main__":
    # Create a logger for a component
    logger = NetdataLogger(component_name="example-component")
    
    # Log some messages
    logger.info("Example component started")
    logger.warning("This is a warning", reason="example", value=42)
    
    # Send some metrics
    logger.gauge("example_value", 123.45)
    logger.counter("example_counter", 1)
    
    # Start system metrics collection
    logger.start_system_metrics(interval=2)
    
    # Simulate some work
    for i in range(10):
        logger.info(f"Processing item {i}")
        logger.gauge("processing_item", i)
        time.sleep(1)
    
    # Stop system metrics collection
    logger.stop_system_metrics()
    
    logger.info("Example component finished")