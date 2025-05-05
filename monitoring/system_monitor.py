"""
System Monitor for NextGen Platform

This module implements a system monitor that collects and sends system metrics
using the SystemMetricsCollector and NetdataLogger.
"""

import time
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

def main():
    """
    Main function to initialize and run the system monitor.
    """
    # Initialize NetdataLogger for the system monitor component
    logger = NetdataLogger(component_name="system-monitor")
    logger.info("System monitor initialized")

    # Initialize and start the SystemMetricsCollector
    metrics_collector = SystemMetricsCollector(logger)
    metrics_collector.start()
    logger.info("System metrics collection started")

    try:
        # Keep the script running for continuous monitoring
        while True:
            time.sleep(60) # Collect metrics every 60 seconds
    except KeyboardInterrupt:
        logger.info("System monitor interrupted. Shutting down.")
    finally:
        # Stop metrics collection on shutdown
        metrics_collector.stop()
        logger.info("System monitor shut down.")

if __name__ == "__main__":
    main()
