#!/usr/bin/env python3
"""
System Metrics Collector Main Script

This script initializes and runs the SystemMetricsCollector to collect
system metrics and expose them through Prometheus.
"""

import os
import time
import logging
import signal
import sys
import psutil
from pathlib import Path

# Make sure we can import modules from the project
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_metrics_main")

# Import the SystemMetricsCollector
from prometheus.system_metrics import SystemMetricsCollector

class SystemMetricsService:
    """
    Service wrapper for the SystemMetricsCollector.
    Handles initialization, running, and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize the service."""
        self.running = False
        self.collector = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def start(self):
        """Start the system metrics collector service."""
        logger.info("Starting System Metrics Collector service...")
        
        # Initialize the collector
        try:
            # Get configuration from environment variables
            interval = int(os.getenv('METRICS_INTERVAL', '5'))
            service_name = os.getenv('METRICS_SERVICE_NAME', 'system_metrics')
            enable_loki = os.getenv('METRICS_ENABLE_LOKI', 'true').lower() == 'true'
            
            # Create the collector
            self.collector = SystemMetricsCollector(
                interval=interval,
                service_name=service_name,
                enable_loki=enable_loki
            )
            
            logger.info(f"SystemMetricsCollector initialized with interval={interval}s")
            
            # Start collecting metrics
            self.running = True
            self.run_metrics_loop()
            
        except Exception as e:
            logger.error(f"Error starting System Metrics Collector: {e}")
            sys.exit(1)
    
    def run_metrics_loop(self):
        """Run the metrics collection loop."""
        logger.info("Starting metrics collection loop...")
        
        while self.running:
            try:
                # Collect CPU metrics
                self.collector.collect_cpu_metrics()
                
                # Collect memory metrics
                self.collector.collect_memory_metrics()
                
                # Collect disk metrics
                self.collector.collect_disk_metrics()
                
                # Collect network metrics
                self.collector.collect_network_metrics()
                
                # Collect GPU metrics if available
                if self.collector.has_gpu:
                    if hasattr(self.collector, 'collect_gpu_metrics_nvml') and hasattr(self.collector, 'NVML_AVAILABLE') and self.collector.NVML_AVAILABLE:
                        self.collector.collect_gpu_metrics_nvml()
                    elif hasattr(self.collector, 'collect_gpu_metrics_gputil'):
                        self.collector.collect_gpu_metrics_gputil()
                
                # Collect system uptime
                uptime = time.time() - psutil.boot_time()
                self.collector.prom.set_gauge("system_uptime_seconds", uptime)
                
                # Collect process count
                process_count = len(psutil.pids())
                self.collector.prom.set_gauge("process_count", process_count, state="total")
                
                # Sleep for the configured interval
                time.sleep(self.collector.interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                # Sleep for a short time to avoid tight error loops
                time.sleep(1)
    
    def stop(self):
        """Stop the service."""
        logger.info("Stopping System Metrics Collector service...")
        self.running = False


if __name__ == "__main__":
    # Create and start the service
    service = SystemMetricsService()
    
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        service.stop()
        logger.info("System Metrics Collector service stopped")
