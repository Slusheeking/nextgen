#!/bin/bash

# Setup Netdata
# This script installs and starts Netdata for monitoring with a custom frontend

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Netdata for NextGen monitoring...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create Netdata data directory if it doesn't exist
NETDATA_DATA_DIR="$PROJECT_DIR/netdata_data"
mkdir -p "$NETDATA_DATA_DIR"
echo -e "${GREEN}Created Netdata data directory at $NETDATA_DATA_DIR${NC}"

# Create Netdata config directory
NETDATA_CONFIG_DIR="$PROJECT_DIR/monitoring/netdata"
mkdir -p "$NETDATA_CONFIG_DIR"

# Check if Netdata is already installed
if command -v netdata &> /dev/null; then
    echo -e "${GREEN}Netdata is already installed${NC}"
else
    # Install Netdata
    echo -e "${YELLOW}Installing Netdata...${NC}"
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    sudo apt-get update
    sudo apt-get install -y curl wget zlib1g-dev uuid-dev libmnl-dev gcc make git autoconf autoconf-archive autogen automake pkg-config python3-pip
    
    # Install Python packages for GPU monitoring
    echo -e "${YELLOW}Installing GPU monitoring tools...${NC}"
    pip3 install --user gputil py3nvml
    
    # Download and run the Netdata installer
    echo -e "${YELLOW}Downloading and running Netdata installer...${NC}"
    bash <(curl -Ss https://my-netdata.io/kickstart.sh) --dont-wait --no-updates --stable-channel --disable-telemetry
    
    # Install additional Netdata plugins for GPU monitoring
    echo -e "${YELLOW}Setting up GPU monitoring for Netdata...${NC}"
    
    # Check if NVIDIA drivers are installed
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA drivers detected, setting up GPU monitoring${NC}"
        
        # Create a custom Python plugin for Netdata to monitor NVIDIA GPUs
        NETDATA_PLUGINS_DIR="/usr/libexec/netdata/python.d"
        if [ ! -d "$NETDATA_PLUGINS_DIR" ]; then
            NETDATA_PLUGINS_DIR="/usr/lib/netdata/python.d"
        fi
        
        if [ -d "$NETDATA_PLUGINS_DIR" ]; then
            # Create NVIDIA GPU plugin
            cat > "$NETDATA_PLUGINS_DIR/nvidia_smi.chart.py" << EOF
#!/usr/bin/env python3

import subprocess
import json
from bases.FrameworkServices.SimpleService import SimpleService

priority = 90000
retries = 60
update_every = 1

ORDER = [
    'utilization',
    'memory',
    'temperature',
    'power',
    'fan'
]

CHARTS = {
    'utilization': {
        'options': [None, 'GPU Utilization', 'percentage', 'utilization', 'nvidia.gpu_utilization', 'line'],
        'lines': []
    },
    'memory': {
        'options': [None, 'Memory Utilization', 'percentage', 'memory', 'nvidia.memory_utilization', 'line'],
        'lines': []
    },
    'temperature': {
        'options': [None, 'GPU Temperature', 'Celsius', 'temperature', 'nvidia.temperature', 'line'],
        'lines': []
    },
    'power': {
        'options': [None, 'Power Utilization', 'Watts', 'power', 'nvidia.power', 'line'],
        'lines': []
    },
    'fan': {
        'options': [None, 'Fan Speed', 'percentage', 'fan', 'nvidia.fan', 'line'],
        'lines': []
    }
}

class Service(SimpleService):
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.gpu_count = 0
        self.gpus = []

    def check(self):
        try:
            nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
            self.gpu_count = nvidia_smi_output.count('GPU')
            
            if self.gpu_count == 0:
                self.error("No NVIDIA GPUs detected")
                return False
                
            # Initialize chart lines based on GPU count
            for i in range(self.gpu_count):
                CHARTS['utilization']['lines'].append(['gpu{0}_utilization'.format(i), 'GPU {0}'.format(i), 'absolute'])
                CHARTS['memory']['lines'].append(['gpu{0}_memory'.format(i), 'GPU {0}'.format(i), 'absolute'])
                CHARTS['temperature']['lines'].append(['gpu{0}_temp'.format(i), 'GPU {0}'.format(i), 'absolute'])
                CHARTS['power']['lines'].append(['gpu{0}_power'.format(i), 'GPU {0}'.format(i), 'absolute'])
                CHARTS['fan']['lines'].append(['gpu{0}_fan'.format(i), 'GPU {0}'.format(i), 'absolute'])
            
            return True
        except Exception as e:
            self.error("Error checking NVIDIA GPUs: {0}".format(str(e)))
            return False

    def get_data(self):
        data = {}
        
        try:
            # Get GPU stats in JSON format
            nvidia_smi_output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed',
                '--format=csv,noheader,nounits'
            ]).decode('utf-8')
            
            for line in nvidia_smi_output.strip().split('\\n'):
                values = line.split(', ')
                if len(values) >= 7:
                    gpu_id = int(values[0])
                    gpu_util = float(values[1])
                    mem_used = float(values[2])
                    mem_total = float(values[3])
                    mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                    temp = float(values[4])
                    power = float(values[5])
                    fan = float(values[6])
                    
                    data['gpu{0}_utilization'.format(gpu_id)] = gpu_util
                    data['gpu{0}_memory'.format(gpu_id)] = mem_util
                    data['gpu{0}_temp'.format(gpu_id)] = temp
                    data['gpu{0}_power'.format(gpu_id)] = power
                    data['gpu{0}_fan'.format(gpu_id)] = fan
            
            return data
        except Exception as e:
            self.error("Error collecting GPU data: {0}".format(str(e)))
            return None
EOF
            
            # Make the plugin executable
            chmod +x "$NETDATA_PLUGINS_DIR/nvidia_smi.chart.py"
            
            # Create configuration file
            mkdir -p /etc/netdata/python.d
            cat > /etc/netdata/python.d/nvidia_smi.conf << EOF
# netdata python.d.plugin configuration for nvidia_smi
#
# This file is in YaML format. Generally the format is:
#
# name: value
#
update_every: 1
priority: 90000
retries: 60
EOF
            
            echo -e "${GREEN}NVIDIA GPU monitoring plugin installed${NC}"
        else
            echo -e "${YELLOW}Could not find Netdata plugins directory, skipping GPU plugin installation${NC}"
        fi
    else
        echo -e "${YELLOW}NVIDIA drivers not detected, skipping GPU monitoring setup${NC}"
    fi
    
    echo -e "${GREEN}Netdata installed successfully${NC}"
fi

# Create custom Netdata configuration
echo -e "${YELLOW}Creating custom Netdata configuration...${NC}"
cat > "$NETDATA_CONFIG_DIR/netdata.conf" << EOF
[global]
    hostname = nextgen-monitoring
    update every = 1
    memory mode = save
    page cache size = 32
    dbengine disk space = 256
    stock config directory = /etc/netdata

[web]
    allow connections from = localhost 127.0.0.1 ::1
    allow dashboard from = localhost 127.0.0.1 ::1
    bind to = 0.0.0.0:19999
EOF
echo -e "${GREEN}Netdata configuration created${NC}"

# Create custom dashboard directory
DASHBOARD_DIR="$PROJECT_DIR/monitoring/dashboard"
mkdir -p "$DASHBOARD_DIR"

# Create a simple custom dashboard
echo -e "${YELLOW}Creating custom dashboard...${NC}"
cat > "$DASHBOARD_DIR/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NextGen Monitoring Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .dashboard-section {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 20px;
        }
        .chart-container {
            width: 100%;
            height: 300px;
            margin-bottom: 20px;
        }
        h2 {
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .netdata-container {
            width: 100%;
            height: 300px;
            margin-bottom: 20px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }
    </style>
    <script type="text/javascript" src="http://localhost:19999/dashboard.js"></script>
</head>
<body>
    <header>
        <h1>NextGen Monitoring Dashboard</h1>
    </header>
    
    <div class="container">
        <div class="dashboard-section">
            <h2>System Overview</h2>
            <div class="dashboard-grid">
                <div class="netdata-container">
                    <div data-netdata="system.cpu"
                         data-title="CPU Usage"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
                <div class="netdata-container">
                    <div data-netdata="system.ram"
                         data-title="Memory Usage"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2>Disk & Network</h2>
            <div class="dashboard-grid">
                <div class="netdata-container">
                    <div data-netdata="disk_space._"
                         data-title="Disk Space Usage"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
                <div class="netdata-container">
                    <div data-netdata="system.io"
                         data-title="Disk I/O"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
                <div class="netdata-container">
                    <div data-netdata="system.net"
                         data-title="Network Traffic"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2>Application Metrics</h2>
            <div class="dashboard-grid">
                <div class="netdata-container">
                    <div data-netdata="apps.cpu"
                         data-title="Application CPU Usage"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
                <div class="netdata-container">
                    <div data-netdata="apps.mem"
                         data-title="Application Memory Usage"
                         data-chart-library="dygraph"
                         data-width="100%"
                         data-height="100%"
                         data-after="-300"
                         data-legend="true"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Refresh charts every 1 second
        NETDATA.options.current.update_every = 1;
    </script>
</body>
</html>
EOF
echo -e "${GREEN}Custom dashboard created${NC}"

# Create a simple web server to serve the dashboard
echo -e "${YELLOW}Creating dashboard server...${NC}"
cat > "$DASHBOARD_DIR/server.py" << EOF
#!/usr/bin/env python3
"""
Simple HTTP server for NextGen monitoring dashboard
"""
import http.server
import socketserver
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving dashboard at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
EOF
chmod +x "$DASHBOARD_DIR/server.py"
echo -e "${GREEN}Dashboard server created${NC}"

# Create a systemd service file for the dashboard
echo -e "${YELLOW}Creating dashboard service...${NC}"
cat > "$NETDATA_CONFIG_DIR/nextgen-dashboard.service" << EOF
[Unit]
Description=NextGen Monitoring Dashboard
After=network.target netdata.service

[Service]
User=ubuntu
ExecStart=/usr/bin/python3 $DASHBOARD_DIR/server.py
WorkingDirectory=$DASHBOARD_DIR
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=nextgen-dashboard
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
echo -e "${GREEN}Dashboard service file created${NC}"

# Copy the service file to systemd directory
echo -e "${YELLOW}Copying service file to systemd directory...${NC}"
sudo cp "$NETDATA_CONFIG_DIR/nextgen-dashboard.service" /etc/systemd/system/
echo -e "${GREEN}Service file copied${NC}"

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
sudo systemctl daemon-reload
echo -e "${GREEN}Systemd reloaded${NC}"

# Enable the dashboard service
echo -e "${YELLOW}Enabling dashboard service...${NC}"
sudo systemctl enable nextgen-dashboard.service
echo -e "${GREEN}Service enabled${NC}"

# Start the dashboard service
echo -e "${YELLOW}Starting dashboard service...${NC}"
sudo systemctl start nextgen-dashboard.service
echo -e "${GREEN}Service started${NC}"

# Check the dashboard service status
echo -e "${YELLOW}Checking dashboard service status...${NC}"
sudo systemctl status nextgen-dashboard.service

# Apply Netdata configuration
echo -e "${YELLOW}Applying Netdata configuration...${NC}"
sudo cp "$NETDATA_CONFIG_DIR/netdata.conf" /etc/netdata/netdata.conf
sudo systemctl restart netdata
echo -e "${GREEN}Netdata configuration applied${NC}"

echo -e "${GREEN}Netdata setup complete!${NC}"
echo -e "${GREEN}Netdata is now running at http://localhost:19999${NC}"
echo -e "${GREEN}Custom dashboard is running at http://localhost:8080${NC}"
echo -e "${GREEN}To check the Netdata status, run: sudo systemctl status netdata${NC}"
echo -e "${GREEN}To check the dashboard status, run: sudo systemctl status nextgen-dashboard.service${NC}"

# Install the monitoring system service for 24/7 operation
echo -e "${YELLOW}Setting up 24/7 monitoring service...${NC}"

# Copy the service file to systemd directory
sudo cp "$PROJECT_DIR/monitoring/netdata/nextgen-monitoring.service" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable nextgen-monitoring.service

# Start the service
sudo systemctl start nextgen-monitoring.service

# Check the service status
sudo systemctl status nextgen-monitoring.service

echo -e "${GREEN}The monitoring system is now running as a 24/7 service${NC}"
echo -e "${GREEN}To check the monitoring service status, run: sudo systemctl status nextgen-monitoring.service${NC}"
echo -e "${GREEN}To view logs, run: sudo journalctl -u nextgen-monitoring.service${NC}"