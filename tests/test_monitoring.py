"""
Test Monitoring System

This script tests the monitoring system by logging messages and checking if they appear in the log files.
"""

import os
import sys
import time
from datetime import datetime

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitoring.system_monitor import MonitoringManager

def main():
    """Main function to test the monitoring system."""
    print("Testing monitoring system...")
    
    # Create a monitoring manager
    monitor = MonitoringManager(service_name="test-service")
    
    # Log some messages
    print("Logging messages...")
    monitor.log_info("This is an info message", component="test", action="test_info")
    monitor.log_warning("This is a warning message", component="test", action="test_warning")
    monitor.log_error("This is an error message", component="test", action="test_error")
    monitor.log_critical("This is a critical message", component="test", action="test_critical")
    
    # Set some metrics
    print("Setting metrics...")
    monitor.set_gauge("test_gauge", 42.0)
    monitor.increment_counter("test_counter")
    monitor.observe_histogram("test_histogram", 0.1)
    monitor.observe_summary("test_summary", 0.2)
    
    # Wait a moment for the logs to be written
    time.sleep(1)
    
    # Check if the log files exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    service_log = os.path.join(logs_dir, "test-service.log")
    master_log = os.path.join(logs_dir, "master.log")
    
    print(f"Checking log files...")
    if os.path.exists(service_log):
        print(f"Service log file exists: {service_log}")
        # Print the last few lines of the service log
        with open(service_log, "r") as f:
            lines = f.readlines()
            print(f"Last few lines of service log:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print(f"Service log file does not exist: {service_log}")
    
    if os.path.exists(master_log):
        print(f"Master log file exists: {master_log}")
        # Print the last few lines of the master log
        with open(master_log, "r") as f:
            lines = f.readlines()
            print(f"Last few lines of master log:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print(f"Master log file does not exist: {master_log}")
    
    print("Testing complete!")

if __name__ == "__main__":
    main()