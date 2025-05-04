"""
Redis Module

This module provides Redis server functionality with Loki logging and Prometheus monitoring.
It's configured to work with the official Redis server.
"""

from redis.redis_server import RedisServer

__all__ = ['RedisServer']
