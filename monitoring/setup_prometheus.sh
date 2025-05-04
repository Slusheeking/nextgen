#!/bin/bash

# Setup Prometheus
# This script installs and starts Prometheus for monitoring

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Prometheus for NextGen monitoring...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create Prometheus data directory if it doesn't exist
PROMETHEUS_DATA_DIR="$PROJECT_DIR/prometheus_data"
mkdir -p "$PROMETHEUS_DATA_DIR"
echo -e "${GREEN}Created Prometheus data directory at $PROMETHEUS_DATA_DIR${NC}"

# Check if Prometheus is already installed
if command -v prometheus &> /dev/null; then
    echo -e "${GREEN}Prometheus is already installed${NC}"
else
    # Install Prometheus
    echo -e "${YELLOW}Installing Prometheus...${NC}"
    
    # Download Prometheus
    PROMETHEUS_VERSION="2.45.0"
    PROMETHEUS_URL="https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    
    # Create a temporary directory
    TMP_DIR=$(mktemp -d)
    
    # Download and extract Prometheus
    echo -e "${YELLOW}Downloading Prometheus...${NC}"
    curl -L "$PROMETHEUS_URL" -o "$TMP_DIR/prometheus.tar.gz"
    
    echo -e "${YELLOW}Extracting Prometheus...${NC}"
    tar -xzf "$TMP_DIR/prometheus.tar.gz" -C "$TMP_DIR"
    
    # Copy Prometheus binaries to /usr/local/bin
    echo -e "${YELLOW}Installing Prometheus binaries...${NC}"
    sudo cp "$TMP_DIR/prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus" /usr/local/bin/
    sudo cp "$TMP_DIR/prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool" /usr/local/bin/
    
    # Clean up
    rm -rf "$TMP_DIR"
    
    echo -e "${GREEN}Prometheus installed successfully${NC}"
fi

# Copy the Prometheus configuration file
echo -e "${YELLOW}Copying Prometheus configuration...${NC}"
mkdir -p "$PROJECT_DIR/monitoring/prometheus"
cat > "$PROJECT_DIR/monitoring/prometheus/prometheus.yml" << EOF
# Global config
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alertmanager configuration
alerting:
  alertmanagers:
  - static_configs:
    - targets:
      # - alertmanager:9093

# Load rules once and periodically evaluate them
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

# A scrape configuration containing exactly one endpoint to scrape:
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
    - targets: ['localhost:9090']

  - job_name: 'nextgen'
    static_configs:
    - targets: ['localhost:8010']
      labels:
        service: 'nextgen'
EOF
echo -e "${GREEN}Prometheus configuration created${NC}"

# Copy the Prometheus service file
echo -e "${YELLOW}Creating Prometheus service...${NC}"
cat > "$PROJECT_DIR/monitoring/prometheus/prometheus.service" << EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=ubuntu
ExecStart=/usr/local/bin/prometheus \\
    --config.file=$PROJECT_DIR/monitoring/prometheus/prometheus.yml \\
    --storage.tsdb.path=$PROMETHEUS_DATA_DIR \\
    --web.console.templates=/etc/prometheus/consoles \\
    --web.console.libraries=/etc/prometheus/console_libraries \\
    --web.listen-address=0.0.0.0:9090 \\
    --web.enable-lifecycle

Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=prometheus
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo -e "${GREEN}Prometheus service file created${NC}"

# Copy the service file to systemd directory
echo -e "${YELLOW}Copying service file to systemd directory...${NC}"
sudo cp "$PROJECT_DIR/monitoring/prometheus/prometheus.service" /etc/systemd/system/
echo -e "${GREEN}Service file copied${NC}"

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
sudo systemctl daemon-reload
echo -e "${GREEN}Systemd reloaded${NC}"

# Enable the service
echo -e "${YELLOW}Enabling prometheus service...${NC}"
sudo systemctl enable prometheus.service
echo -e "${GREEN}Service enabled${NC}"

# Start the service
echo -e "${YELLOW}Starting prometheus service...${NC}"
sudo systemctl start prometheus.service
echo -e "${GREEN}Service started${NC}"

# Check the service status
echo -e "${YELLOW}Checking service status...${NC}"
sudo systemctl status prometheus.service

echo -e "${GREEN}Prometheus setup complete!${NC}"
echo -e "${GREEN}Prometheus is now running at http://localhost:9090${NC}"
echo -e "${GREEN}To check the service status, run: sudo systemctl status prometheus.service${NC}"
echo -e "${GREEN}To stop the service, run: sudo systemctl stop prometheus.service${NC}"
echo -e "${GREEN}To restart the service, run: sudo systemctl restart prometheus.service${NC}"