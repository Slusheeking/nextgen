"""
Redis Module

This module provides Redis server functionality with integrated monitoring using NetdataLogger.
It's configured to work with the official Redis server and provides a centralized Redis
instance for the NextGen FinGPT system. The Redis server is implemented as a singleton
to ensure only one instance is running at a time.

Features:
- Centralized Redis server instance
- Integrated monitoring with NetdataLogger
- Configuration via JSON files or environment variables
- Automatic metrics collection
"""

# Import the RedisServer class for easier access
from .redis_server import RedisServer

__all__ = ["RedisServer"]
