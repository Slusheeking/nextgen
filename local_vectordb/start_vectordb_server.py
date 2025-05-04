#!/usr/bin/env python3
"""
Vector DB Server Starter

This script starts the Vector Database server with Loki logging and Prometheus monitoring integration.
It is designed to be used as the entry point for the systemd service.
"""

import os
import sys
import time
import logging
import signal

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Vector DB server
from local_vectordb.vectordb_server import VectorDBServer
# Import monitoring utilities
from monitoring import setup_monitoring


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vectordb_starter")



# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="local_vectordb-start-vectordb-server",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "local_vectordb/vectordb_server"}
)


def handle_signal(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point for the Vector DB server starter."""
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Starting Vector DB server with Loki and Prometheus integration...")

    try:
        # Get Vector DB server instance
        vector_db_server = VectorDBServer.get_instance()

        # Get Vector DB client
        client = vector_db_server.get_client()

        if client:
            # Test connection by listing collections
            collections = vector_db_server.list_collections()
            logger.info(f"Vector DB server started successfully with {len(collections)} collections")
            
            # Create default collection if it doesn't exist
            default_collection_name = vector_db_server.default_collection_name
            if default_collection_name not in collections:
                logger.info(f"Creating default collection: {default_collection_name}")
                vector_db_server.get_collection(default_collection_name)

            # Keep the server running
            while True:
                time.sleep(10)

                # Periodically check if Vector DB is still running
                try:
                    # Just list collections to verify connection is alive
                    vector_db_server.list_collections()
                except Exception as e:
                    logger.error(f"Vector DB connection lost: {e}")
                    sys.exit(1)
        else:
            logger.error("Failed to get Vector DB client")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error starting Vector DB server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()