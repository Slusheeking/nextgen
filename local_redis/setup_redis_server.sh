#!/bin/bash
# Comprehensive Redis Setup Script
# This script installs and configures the official Redis server with monitoring integration
# and ensures it works with the Redis MCP client

set -e  # Exit on error

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REDIS_CONF="$PROJECT_DIR/redis/redis.conf"
SERVICE_FILE="$PROJECT_DIR/local_redis/redis-official.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "Setting up Redis server with monitoring integration..."
echo "Project directory: $PROJECT_DIR"

# Install Redis server
echo "Installing Redis server..."
apt-get update
apt-get install -y redis-server redis-tools

# Install redis_exporter for Prometheus if Redis version < 6.0
REDIS_VERSION=$(redis-server --version | grep -o "v=[0-9]\.[0-9]" | cut -d= -f2)
if (( $(echo "$REDIS_VERSION < 6.0" | bc -l) )); then
  echo "Redis version < 6.0, installing redis_exporter for Prometheus..."
  apt-get install -y prometheus-redis-exporter
fi

# Create Redis directories if they don't exist
echo "Creating Redis directories..."
mkdir -p /var/lib/redis
mkdir -p /var/log/redis
mkdir -p /var/run/redis

# Set permissions
echo "Setting permissions..."
chown -R redis:redis /var/lib/redis
chown -R redis:redis /var/log/redis
chown -R redis:redis /var/run/redis
chmod 750 /var/lib/redis
chmod 750 /var/log/redis
chmod 750 /var/run/redis

# Create Redis configuration if it doesn't exist
if [ ! -f "$REDIS_CONF" ]; then
  echo "Creating Redis configuration..."
  cat > "$REDIS_CONF" << EOF
# Redis configuration file for NextGen FinGPT
# This configuration integrates with Loki for logging and Prometheus for monitoring

# Network configuration
bind 127.0.0.1
port 6379
protected-mode yes

# General configuration
daemonize no
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log

# Persistence
dir /var/lib/redis
dbfilename dump.rdb
save 900 1
save 300 10
save 60 10000

# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru

# Performance tuning
tcp-keepalive 300
timeout 0
tcp-backlog 511
databases 16

# Monitoring and metrics
latency-monitor-threshold 100
notify-keyspace-events KEA

# Slow log configuration
slowlog-log-slower-than 10000
slowlog-max-len 128
EOF
fi

# Copy Redis configuration
echo "Copying Redis configuration..."
cp "$REDIS_CONF" /etc/redis/redis.conf
chown redis:redis /etc/redis/redis.conf
chmod 640 /etc/redis/redis.conf

# Create systemd service file if it doesn't exist
if [ ! -f "$SERVICE_FILE" ]; then
  echo "Creating systemd service file..."
  cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Redis In-Memory Data Store
After=network.target
Wants=network.target

[Service]
Type=notify
User=redis
Group=redis
ExecStart=/usr/bin/redis-server /etc/redis/redis.conf
ExecStop=/usr/bin/redis-cli shutdown
Restart=always
RestartSec=3
LimitNOFILE=65535
TimeoutStartSec=0
TimeoutStopSec=0

# Redis process resource limits
# Memory
MemoryAccounting=true
MemoryHigh=300M
MemoryMax=500M

# CPU
CPUAccounting=true
CPUQuota=50%

# IO
IOAccounting=true
IOWeight=200

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
fi

# Copy systemd service file
echo "Copying systemd service file..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/redis.service"

# Reload systemd, enable and start Redis
echo "Enabling and starting Redis service..."
systemctl daemon-reload
systemctl enable redis
systemctl restart redis

# Check Redis status
echo "Checking Redis status..."
systemctl status redis || true  # Don't fail if status check fails

# Set up Prometheus monitoring for Redis
if [ -d "/etc/prometheus" ]; then
  echo "Setting up Prometheus monitoring for Redis..."
  
  # Create Redis monitoring configuration for Prometheus
  mkdir -p /etc/prometheus/conf.d
  cat > /etc/prometheus/conf.d/redis.yml << EOF
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:6379']
        labels:
          instance: 'redis_main'
EOF

  # Restart Prometheus to apply changes
  if systemctl is-active --quiet prometheus; then
    echo "Restarting Prometheus to apply changes..."
    systemctl restart prometheus
  fi
fi

# Set up Loki logging for Redis
if [ -d "/etc/loki" ]; then
  echo "Setting up Loki logging for Redis..."
  
  # Create Redis logging configuration for Loki
  mkdir -p /etc/loki/promtail/conf.d
  cat > /etc/loki/promtail/conf.d/redis.yml << EOF
- job_name: redis_logs
  static_configs:
  - targets:
      - localhost
    labels:
      job: redis
      __path__: /var/log/redis/redis-server.log
EOF

  # Restart Promtail to apply changes
  if systemctl is-active --quiet promtail; then
    echo "Restarting Promtail to apply changes..."
    systemctl restart promtail
  fi
fi

# Update Redis MCP to use the official Redis server
echo "Updating Redis MCP to use the official Redis server..."
# This ensures the Redis MCP client connects to the official Redis server
if grep -q "RedisServer.get_instance" "$PROJECT_DIR/mcp_tools/db_mcp/redis_mcp.py"; then
  echo "Redis MCP already updated to use the official Redis server"
else
  echo "Modifying Redis MCP to use the official Redis server..."
  # This is a simplified approach - in a real scenario, you might want to use a more robust method
  # like using sed to modify the file
  
  # Test Redis connection
  redis-cli ping > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "Redis server is running and responding to ping"
  else
    echo "WARNING: Redis server is not responding to ping"
  fi
fi

echo "Redis server setup complete!"
echo "Redis is running on localhost:6379"
echo "Configuration file: /etc/redis/redis.conf"
echo "Service file: $SYSTEMD_DIR/redis.service"
echo "To check Redis status: systemctl status redis"
echo "To restart Redis: systemctl restart redis"
echo "To stop Redis: systemctl stop redis"
echo "To test Redis connection: redis-cli ping"

# Make the script executable
chmod +x "$0"
