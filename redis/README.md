# Redis Server Setup for NextGen FinGPT

This directory contains the necessary files to set up and configure the official Redis server with Loki logging and Prometheus monitoring integration for the NextGen FinGPT system.

## Overview

The Redis setup includes:

1. **Official Redis Server**: A high-performance, in-memory data store that runs 24/7 in production environments
2. **Monitoring Integration**: Connects Redis with Loki for logging and Prometheus for metrics
3. **Redis MCP Client**: Provides Redis functionality to the NextGen FinGPT system through the MCP protocol

## Files

- `redis.conf`: Redis server configuration file
- `redis-official.service`: Systemd service file for the Redis server
- `setup_redis_server.sh`: Script to install and configure Redis
- `test_redis_setup.py`: Script to test the Redis setup
- `redis_server.py`: Python wrapper for Redis (used during development)

## Installation

To install and configure the Redis server:

```bash
# Make the setup script executable
chmod +x setup_redis_server.sh

# Run the setup script as root
sudo ./setup_redis_server.sh
```

This script will:
1. Install the official Redis server
2. Configure it with monitoring integration
3. Set up the systemd service
4. Ensure it connects with the Redis MCP client

## Testing

To verify that the Redis setup works correctly:

```bash
# Make the test script executable
chmod +x test_redis_setup.py

# Run the test script
python3 test_redis_setup.py
```

This will test:
1. Redis connection
2. Basic Redis operations (set, get, delete)
3. Redis JSON operations
4. Redis MCP direct methods

## Usage

The Redis server runs automatically as a systemd service. You can manage it using the following commands:

```bash
# Check Redis status
sudo systemctl status redis

# Restart Redis
sudo systemctl restart redis

# Stop Redis
sudo systemctl stop redis

# View Redis logs
sudo journalctl -u redis
```

## Redis MCP Client

The Redis MCP client is automatically configured to connect to the Redis server. You can use it in your code like this:

```python
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# Create Redis MCP client
redis_mcp = RedisMCP()

# Use Redis operations
redis_mcp.set_value("my_key", "my_value")
value = redis_mcp.get_value("my_key")
```

## Monitoring

### Prometheus Metrics

Redis metrics are exposed to Prometheus and can be viewed in Grafana dashboards. The metrics include:

- Connected clients
- Memory usage
- Command statistics
- Keyspace statistics

### Loki Logs

Redis logs are forwarded to Loki and can be viewed in Grafana. The logs include:

- Server startup/shutdown
- Client connections
- Command execution
- Errors and warnings

## Configuration

The Redis configuration can be modified in `/etc/redis/redis.conf`. After making changes, restart the Redis server:

```bash
sudo systemctl restart redis
```

Key configuration parameters:

- `bind`: Network interface to bind to (default: 127.0.0.1)
- `port`: Port to listen on (default: 6379)
- `maxmemory`: Maximum memory usage (default: 256MB)
- `maxmemory-policy`: Eviction policy (default: allkeys-lru)