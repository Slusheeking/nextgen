#!/bin/bash

# Start the entire NextGen monitoring system
# This script starts all monitoring components and ensures they're running properly

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting NextGen Monitoring System...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory if it doesn't exist
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Create charts directory if it doesn't exist
CHARTS_DIR="$SCRIPT_DIR/dashboard/charts"
mkdir -p "$CHARTS_DIR"

# Function to check if a service is running
check_service() {
    if systemctl is-active --quiet $1; then
        echo -e "${GREEN}$1 is running${NC}"
        return 0
    else
        echo -e "${RED}$1 is not running${NC}"
        return 1
    fi
}

# Function to start a service if it's not running
start_service() {
    if ! check_service $1; then
        echo -e "${YELLOW}Starting $1...${NC}"
        sudo systemctl start $1
        if check_service $1; then
            echo -e "${GREEN}$1 started successfully${NC}"
        else
            echo -e "${RED}Failed to start $1${NC}"
            return 1
        fi
    fi
    return 0
}

# Check and start Netdata
start_service netdata || {
    echo -e "${YELLOW}Netdata not found or failed to start. Installing...${NC}"
    "$SCRIPT_DIR/setup_netdata.sh"
}

# Check and start log server
start_service log-server.service || {
    echo -e "${YELLOW}Log server not found or failed to start. Installing...${NC}"
    "$SCRIPT_DIR/setup_log_server.sh"
}

# Start the dashboard server
echo -e "${YELLOW}Starting dashboard server...${NC}"
cd "$SCRIPT_DIR/dashboard"
python3 server.py &
DASHBOARD_PID=$!
echo -e "${GREEN}Dashboard server started with PID $DASHBOARD_PID${NC}"

# Write the dashboard PID to a file for later reference
echo $DASHBOARD_PID > "$SCRIPT_DIR/dashboard.pid"

# Monitor the dashboard server and restart if it crashes
(
    while true; do
        if ! ps -p $DASHBOARD_PID > /dev/null; then
            echo -e "${RED}Dashboard server crashed. Restarting...${NC}" >> "$LOGS_DIR/monitoring.log"
            cd "$SCRIPT_DIR/dashboard"
            python3 server.py &
            DASHBOARD_PID=$!
            echo $DASHBOARD_PID > "$SCRIPT_DIR/dashboard.pid"
            echo -e "${GREEN}Dashboard server restarted with PID $DASHBOARD_PID${NC}" >> "$LOGS_DIR/monitoring.log"
        fi
        sleep 10
    done
) &

# Start system metrics collector
echo -e "${YELLOW}Starting system metrics collector...${NC}"
python3 -c "
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

logger = NetdataLogger('system-metrics')
collector = SystemMetricsCollector(logger)
collector.start()

# Keep the script running
import time
while True:
    time.sleep(60)
    logger.info('System metrics collector is running')
" &
METRICS_PID=$!
echo -e "${GREEN}System metrics collector started with PID $METRICS_PID${NC}"

# Write the metrics collector PID to a file for later reference
echo $METRICS_PID > "$SCRIPT_DIR/metrics.pid"

echo -e "${GREEN}NextGen Monitoring System started successfully${NC}"
echo -e "${GREEN}Dashboard is available at http://localhost:8080${NC}"
echo -e "${GREEN}Netdata is available at http://localhost:19999${NC}"
echo -e "${GREEN}Log server is available at http://localhost:8011${NC}"

# Keep the script running to maintain the child processes
while true; do
    sleep 60
    # Check if all components are still running
    if ! ps -p $DASHBOARD_PID > /dev/null; then
        echo -e "${RED}Dashboard server crashed and failed to restart${NC}" >> "$LOGS_DIR/monitoring.log"
    fi
    if ! ps -p $METRICS_PID > /dev/null; then
        echo -e "${RED}System metrics collector crashed${NC}" >> "$LOGS_DIR/monitoring.log"
        # Restart metrics collector
        python3 -c "
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

logger = NetdataLogger('system-metrics')
collector = SystemMetricsCollector(logger)
collector.start()

# Keep the script running
import time
while True:
    time.sleep(60)
    logger.info('System metrics collector is running')
" &
        METRICS_PID=$!
        echo $METRICS_PID > "$SCRIPT_DIR/metrics.pid"
        echo -e "${GREEN}System metrics collector restarted with PID $METRICS_PID${NC}" >> "$LOGS_DIR/monitoring.log"
    fi
done