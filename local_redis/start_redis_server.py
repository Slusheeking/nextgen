#!/usr/bin/env python3
"""
Redis Server Starter

This script starts the Redis server with Loki logging and Prometheus monitoring integration.
It is designed to be used as the entry point for the systemd service.
"""

import os
import sys
import time
import logging
import signal

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Redis server
from local_redis.redis_server import RedisServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis_starter")

def handle_signal(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point for the Redis server starter."""
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    logger.info("Starting Redis server with Loki and Prometheus integration...")
    
    try:
        # Get Redis server instance
        redis_server = RedisServer.get_instance()
        
        # Get Redis client
        redis_client = redis_server.get_client()
        
        if redis_client:
            # Test connection
            redis_client.ping()
            logger.info("Redis server started successfully")
            
            # Keep the server running
            while True:
                time.sleep(10)
                
                # Periodically check if Redis is still running
                try:
                    redis_client.ping()
                except Exception as e:
                    logger.error(f"Redis connection lost: {e}")
                    sys.exit(1)
        else:
            logger.error("Failed to get Redis client")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error starting Redis server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
