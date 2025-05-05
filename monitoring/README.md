# NextGen Monitoring System

A comprehensive monitoring system for the NextGen platform that provides real-time metrics, logs, and financial charts in a unified dashboard.

## Features

- **24/7 Monitoring**: Runs as a system service with automatic restart on failure
- **System Metrics**: CPU, memory, disk, and network usage
- **GPU Monitoring**: NVIDIA GPU metrics (utilization, memory, temperature)
- **Application Metrics**: Process-level metrics for NextGen components
- **Financial Charts**: Real-time and historical stock charts
- **Unified Dashboard**: Single interface for all monitoring needs
- **Logging Integration**: Centralized log collection and viewing

## Components

1. **Netdata**: Core metrics collection and visualization
2. **Custom Dashboard**: Web interface for NextGen-specific metrics and charts
3. **Stock Charts**: Financial data visualization
4. **Log Server**: Centralized log collection and viewing
5. **System Metrics Collector**: Enhanced system metrics collection

## Installation

Run the setup script to install and configure the monitoring system:

```bash
cd /path/to/nextgen
bash monitoring/setup_netdata.sh
```

This will:
- Install Netdata
- Configure GPU monitoring (if NVIDIA GPUs are detected)
- Set up the custom dashboard
- Install the 24/7 monitoring service
- Start all components

## Accessing the Dashboard

After installation, the monitoring system is available at:

- **Main Dashboard**: http://localhost:8080
- **Netdata**: http://localhost:19999
- **Log Server**: http://localhost:8011

## Using Stock Charts

The stock charts feature allows you to:

1. **View Individual Stocks**:
   - Enter a stock symbol (e.g., AAPL)
   - Select a timeframe (1d, 5d, 1m, 3m, 1y)
   - Toggle volume and technical indicators
   - Click "Load Chart" to generate the chart

2. **Compare Multiple Stocks**:
   - Enter comma-separated symbols (e.g., AAPL,MSFT,GOOGL)
   - Select a timeframe
   - Toggle price normalization
   - Click "Compare Stocks" to generate the comparison chart

## System Requirements

- Linux-based operating system
- Python 3.6+
- 2GB+ RAM
- 1GB+ free disk space
- NVIDIA drivers (optional, for GPU monitoring)

## Service Management

The monitoring system runs as a systemd service for 24/7 operation:

```bash
# Check status
sudo systemctl status nextgen-monitoring.service

# Start service
sudo systemctl start nextgen-monitoring.service

# Stop service
sudo systemctl stop nextgen-monitoring.service

# Restart service
sudo systemctl restart nextgen-monitoring.service

# View logs
sudo journalctl -u nextgen-monitoring.service
```

Individual components can also be managed separately:

```bash
# Netdata
sudo systemctl status netdata
sudo systemctl restart netdata

# Dashboard
sudo systemctl status nextgen-dashboard.service
sudo systemctl restart nextgen-dashboard.service

# Log server
sudo systemctl status log-server.service
sudo systemctl restart log-server.service
```

## Exported Metrics

The monitoring system exports the following metrics:

### System Metrics
- CPU usage (overall and per-core)
- Memory usage (RAM, swap)
- Disk usage and I/O
- Network traffic
- System load
- Process metrics

### GPU Metrics (if available)
- GPU utilization
- GPU memory usage
- GPU temperature
- GPU power consumption
- Fan speed

### Application Metrics
- Process CPU usage
- Process memory usage
- Process disk I/O
- Process network I/O
- Custom application metrics

## Troubleshooting

If you encounter issues with the monitoring system:

1. **Check service status**:
   ```bash
   sudo systemctl status nextgen-monitoring.service
   ```

2. **View logs**:
   ```bash
   sudo journalctl -u nextgen-monitoring.service
   ```

3. **Restart the service**:
   ```bash
   sudo systemctl restart nextgen-monitoring.service
   ```

4. **Check individual components**:
   ```bash
   sudo systemctl status netdata
   sudo systemctl status nextgen-dashboard.service
   sudo systemctl status log-server.service
   ```

5. **Verify Netdata is running**:
   ```bash
   curl http://localhost:19999
   ```

## Extending the Monitoring System

The monitoring system can be extended with:

1. **Custom Metrics**: Add new metrics to the `system_metrics.py` file
2. **Dashboard Panels**: Add new panels to the dashboard in `dashboard/index.html`
3. **Additional Charts**: Add new chart types to `stock_charts.py`

## License

This monitoring system is part of the NextGen platform and is subject to the same license terms.