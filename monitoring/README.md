# NextGen Monitoring System

This directory contains a complete monitoring setup for the NextGen platform using:
- **Prometheus**: For metrics collection and storage
- **File-based Logging**: For log aggregation and storage
- **Log API Server**: For accessing logs via a REST API

Both services run as systemd services that start automatically on boot and run 24/7.

## Setup

To install both Prometheus and the Log API Server, simply run:

```bash
./setup_monitoring_system.sh
```

This script will:
1. Download and install Prometheus
2. Set up the Log API Server
3. Configure them with the provided configuration files
4. Set up systemd services to run them 24/7
5. Start the services

## Services

### Prometheus
- **Port**: 9090
- **UI URL**: http://localhost:9090
- **Service control**: `sudo systemctl [start|stop|restart|status] prometheus`
- **Configuration**: `monitoring/prometheus/prometheus.yml`

### Log API Server
- **Port**: 8011
- **API URL**: http://localhost:8011
- **Service control**: `sudo systemctl [start|stop|restart|status] log-server`
- **Log files**: `logs/` directory

## Integration with system_monitor.py

The `system_monitor.py` module is already configured to work with Prometheus and file-based logging:

- It exposes metrics on port 8010 for Prometheus to scrape
- It writes logs to files in the `logs/` directory
- It writes to a master log file (`logs/master.log`) for centralized logging

Example usage:

```python
from monitoring.system_monitor import MonitoringManager, SystemMetricsCollector

# Initialize the monitoring manager
monitor = MonitoringManager(service_name="my-service")

# Log something (goes to log files)
monitor.log_info("Service started", component="service", action="startup")

# Update a metric (available to Prometheus)
monitor.set_gauge("active_connections", 5)

# Start system metrics collection (CPU, memory, disk)
collector = SystemMetricsCollector(monitor)
collector.start()
```

## Checking Logs and Metrics

### View Metrics in Prometheus
1. Open http://localhost:9090 in a browser
2. Use the query interface to explore available metrics
3. Example queries:
   - `my_service_cpu_percent` - CPU usage
   - `my_service_memory_percent` - Memory usage
   - `my_service_uptime_seconds` - Service uptime

### View Logs via API

The Log API Server provides several endpoints for accessing logs:

- `GET /logs` - Get a list of available log files
- `GET /logs/{filename}` - Get the content of a specific log file
- `GET /logs/{filename}/stream` - Stream updates to a log file using Server-Sent Events (SSE)
- `GET /logs/master` - Get the content of the master log file
- `GET /logs/master/stream` - Stream updates to the master log file

Example usage:

```javascript
// Get a list of log files
fetch('http://localhost:8011/logs')
  .then(response => response.json())
  .then(data => console.log(data));

// Get the content of the master log file
fetch('http://localhost:8011/logs/master')
  .then(response => response.json())
  .then(data => console.log(data.content));

// Stream updates to the master log file
const eventSource = new EventSource('http://localhost:8011/logs/master/stream');
eventSource.onmessage = event => {
  const data = JSON.parse(event.data);
  console.log(data.line);
};
```

### View Logs Directly

You can also view the log files directly in the `logs/` directory:

```bash
# View the master log file
cat logs/master.log

# View a specific service's log file
cat logs/my-service.log

# Tail the master log file
tail -f logs/master.log
```

## File Structure

- `prometheus/`
  - `prometheus.yml` - Prometheus configuration
  - `prometheus.service` - Systemd service file
- `log_api.py` - Log API Server implementation
- `start_log_server.py` - Script to start the Log API Server
- `log-server.service` - Systemd service file for the Log API Server
- `setup_log_server.sh` - Log API Server installation script
- `setup_prometheus.sh` - Prometheus installation script
- `setup_monitoring_system.sh` - Master setup script
- `system_monitor.py` - NextGen monitoring library
- `fix_loki_endpoint.py` - Script to remove Loki handlers (transitional)
- `LOGGING.md` - Detailed documentation on the logging system

## Additional Documentation

For more detailed information about the logging system, see [LOGGING.md](LOGGING.md).
