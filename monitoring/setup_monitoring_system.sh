#!/bin/bash

# Setup Monitoring System
# This script sets up the entire monitoring system, including:
# 1. Log server for file-based logging
# 2. Prometheus for metrics collection

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up NextGen Monitoring System...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory if it doesn't exist
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"
echo -e "${GREEN}Created logs directory at $LOGS_DIR${NC}"

# Run the log server setup script
echo -e "${YELLOW}Setting up log server...${NC}"
"$SCRIPT_DIR/setup_log_server.sh"
echo -e "${GREEN}Log server setup complete${NC}"

# Run the Prometheus setup script
echo -e "${YELLOW}Setting up Prometheus...${NC}"
"$SCRIPT_DIR/setup_prometheus.sh"
echo -e "${GREEN}Prometheus setup complete${NC}"

echo -e "${GREEN}NextGen Monitoring System setup complete!${NC}"
echo -e "${GREEN}The log server is running at http://localhost:8011${NC}"
echo -e "${GREEN}Prometheus is running at http://localhost:9090${NC}"
echo -e "${GREEN}You can view the logs at $LOGS_DIR${NC}"
echo -e "${GREEN}To check the log server status, run: sudo systemctl status log-server.service${NC}"
echo -e "${GREEN}To check the Prometheus status, run: sudo systemctl status prometheus.service${NC}"