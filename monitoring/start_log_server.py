"""
Start Log API Server

This script starts the FastAPI server that serves log files to the front end.
It ensures the logs directory exists and creates a master log file if needed.

Usage:
    python start_log_server.py
"""

import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import uvicorn
import subprocess

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up a logger for this script
logger = logging.getLogger("start_log_server")
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Add file handler
file_handler = RotatingFileHandler(os.path.join(logs_dir, "log_server.log"), maxBytes=5*1024*1024, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def main():
    """Main function to start the Log API server."""
    logger.info("Starting Log API server")

    
    # Create a master log file if it doesn't exist
    master_log = os.path.join(logs_dir, "master.log")
    if not os.path.exists(master_log):
        with open(master_log, "w") as f:
            f.write(f"--- Master log created at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        logger.info(f"Created master log file at {master_log}")
    
    # Start the FastAPI server
    logger.info("Starting FastAPI server on port 8011")
    uvicorn.run(
        "monitoring.log_api:app",
        host="0.0.0.0",
        port=8011,
        reload=True,
    )

if __name__ == "__main__":
    main()