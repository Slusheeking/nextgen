#!/bin/bash

# Setup Log Server
# This script installs and starts the log server service

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up NextGen Log Server...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory if it doesn't exist
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"
echo -e "${GREEN}Created logs directory at $LOGS_DIR${NC}"

# Copy the service file to systemd directory
echo -e "${YELLOW}Copying service file to systemd directory...${NC}"
sudo cp "$SCRIPT_DIR/log-server.service" /etc/systemd/system/
echo -e "${GREEN}Service file copied${NC}"

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
sudo systemctl daemon-reload
echo -e "${GREEN}Systemd reloaded${NC}"

# Enable the service
echo -e "${YELLOW}Enabling log-server service...${NC}"
sudo systemctl enable log-server.service
echo -e "${GREEN}Service enabled${NC}"

# Start the service
echo -e "${YELLOW}Starting log-server service...${NC}"
sudo systemctl start log-server.service
echo -e "${GREEN}Service started${NC}"

# Check the service status
echo -e "${YELLOW}Checking service status...${NC}"
sudo systemctl status log-server.service

echo -e "${GREEN}Log server setup complete!${NC}"
echo -e "${GREEN}The log server is now running at http://localhost:8011${NC}"
echo -e "${GREEN}You can view the logs at $LOGS_DIR${NC}"
echo -e "${GREEN}To check the service status, run: sudo systemctl status log-server.service${NC}"
echo -e "${GREEN}To stop the service, run: sudo systemctl stop log-server.service${NC}"
echo -e "${GREEN}To restart the service, run: sudo systemctl restart log-server.service${NC}"