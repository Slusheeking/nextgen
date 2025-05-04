#!/bin/bash
# Setup script for the System Metrics Collector service

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up System Metrics Collector service..."

# Install required Python packages
echo "Installing required Python packages..."
pip3 install prometheus_client psutil python-dotenv logging-loki

# Try to install GPU monitoring packages
echo "Attempting to install GPU monitoring packages..."
pip3 install gputil py3nvml pynvml || echo "GPU monitoring packages installation failed, continuing without them"

# Ensure the system_metrics directory has an __init__.py file
if [ ! -f "$SCRIPT_DIR/system_metrics/__init__.py" ]; then
  echo "Creating system_metrics/__init__.py..."
  mkdir -p "$SCRIPT_DIR/system_metrics"
  touch "$SCRIPT_DIR/system_metrics/__init__.py"
fi

# Copy the service file to systemd directory
echo "Installing systemd service file..."
cp "$SCRIPT_DIR/system-metrics.service" /etc/systemd/system/

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable and start the service
echo "Enabling and starting the service..."
systemctl enable system-metrics.service
systemctl start system-metrics.service

# Check the service status
echo "Service status:"
systemctl status system-metrics.service

echo "System Metrics Collector service setup complete!"
echo "You can check the logs with: journalctl -u system-metrics.service -f"
