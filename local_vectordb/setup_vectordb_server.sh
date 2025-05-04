#!/bin/bash
# Vector Database Setup Script
# This script installs and configures the ChromaDB vector database server with monitoring integration
# and ensures it works with the VectorDBMCP client

set -e  # Exit on error

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_FILE="$PROJECT_DIR/local_vectordb/vectordb-server.service"
SYSTEMD_DIR="/etc/systemd/system"
STORAGE_DIR="$PROJECT_DIR/local_vectordb/vector_db_storage"
ENV_FILE="$PROJECT_DIR/.env"

echo "Setting up Vector Database server with monitoring integration..."
echo "Project directory: $PROJECT_DIR"

# Create or update .env file with vector database configuration
if [ -f "$ENV_FILE" ]; then
  echo "Updating existing .env file with vector database settings..."
  # Check if variables already exist in .env, add if not
  grep -q "VECTORDB_PATH" "$ENV_FILE" || echo "VECTORDB_PATH=$STORAGE_DIR" >> "$ENV_FILE"
  grep -q "VECTORDB_HOST" "$ENV_FILE" || echo "VECTORDB_HOST=localhost" >> "$ENV_FILE"
  grep -q "VECTORDB_PORT" "$ENV_FILE" || echo "VECTORDB_PORT=8000" >> "$ENV_FILE"
  grep -q "VECTORDB_USE_HTTP_SERVER" "$ENV_FILE" || echo "VECTORDB_USE_HTTP_SERVER=False" >> "$ENV_FILE"
  
  # Add monitoring configuration with safe defaults
  grep -q "LOKI_URL" "$ENV_FILE" || echo "LOKI_URL=http://localhost:3100" >> "$ENV_FILE"
  grep -q "ENABLE_LOKI" "$ENV_FILE" || echo "ENABLE_LOKI=False" >> "$ENV_FILE"
  grep -q "ENABLE_PROMETHEUS" "$ENV_FILE" || echo "ENABLE_PROMETHEUS=False" >> "$ENV_FILE"
  grep -q "PROMETHEUS_METRICS_PORT" "$ENV_FILE" || echo "PROMETHEUS_METRICS_PORT=8011" >> "$ENV_FILE"
else
  echo "Creating .env file with vector database settings..."
  cat > "$ENV_FILE" << EOF
# Vector Database Configuration
VECTORDB_PATH=$STORAGE_DIR
VECTORDB_HOST=localhost
VECTORDB_PORT=8000
VECTORDB_USE_HTTP_SERVER=False

# Monitoring Configuration
LOKI_URL=http://localhost:3100
ENABLE_LOKI=False
ENABLE_PROMETHEUS=False
PROMETHEUS_METRICS_PORT=8011
EOF
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install chromadb numpy sentence-transformers

# Create vector database directories if they don't exist
echo "Creating vector database directories..."
mkdir -p "$STORAGE_DIR"

# Set permissions
echo "Setting permissions..."
chown -R ubuntu:ubuntu "$STORAGE_DIR"
chmod 750 "$STORAGE_DIR"

# Copy systemd service file
echo "Copying systemd service file..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/vectordb.service"

# Reload systemd, enable and start Vector DB service
echo "Enabling and starting Vector DB service..."
systemctl daemon-reload
systemctl enable vectordb
systemctl restart vectordb

# Check Vector DB status
echo "Checking Vector DB status..."
systemctl status vectordb || true  # Don't fail if status check fails

# Set up Prometheus monitoring for Vector DB
if [ -d "/etc/prometheus" ]; then
  echo "Setting up Prometheus monitoring for Vector DB..."
  
  # Create Vector DB monitoring configuration for Prometheus
  mkdir -p /etc/prometheus/conf.d
  cat > /etc/prometheus/conf.d/vectordb.yml << EOF
  - job_name: 'vectordb'
    static_configs:
      - targets: ['localhost:8011']
        labels:
          instance: 'vectordb_main'
EOF

  # Restart Prometheus to apply changes
  if systemctl is-active --quiet prometheus; then
    echo "Restarting Prometheus to apply changes..."
    systemctl restart prometheus
  fi
fi

# Set up Loki logging for Vector DB
if [ -d "/etc/loki" ]; then
  echo "Setting up Loki logging for Vector DB..."
  
  # Create Vector DB logging configuration for Loki
  mkdir -p /etc/loki/promtail/conf.d
  cat > /etc/loki/promtail/conf.d/vectordb.yml << EOF
- job_name: vectordb_logs
  static_configs:
  - targets:
      - localhost
    labels:
      job: vectordb
      __path__: /var/log/syslog
      __systemd_unit__: vectordb.service
EOF

  # Restart Promtail to apply changes
  if systemctl is-active --quiet promtail; then
    echo "Restarting Promtail to apply changes..."
    systemctl restart promtail
  fi
fi

echo "Vector Database server setup complete!"
echo "Vector DB is running as a systemd service"
echo "Configuration directories:"
echo "  - Storage: $STORAGE_DIR"
echo "Service file: $SYSTEMD_DIR/vectordb.service"
echo "To check Vector DB status: systemctl status vectordb"
echo "To restart Vector DB: systemctl restart vectordb"
echo "To stop Vector DB: systemctl stop vectordb"

# Make the script executable
chmod +x "$0"