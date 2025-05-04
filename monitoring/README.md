# NextGen FinGPT Monitoring Module

This module provides comprehensive monitoring capabilities for the NextGen FinGPT project, including Prometheus metrics collection, Loki logging, and system metrics monitoring.

## Features

- **Unified Monitoring Interface**: A single interface for both Prometheus metrics and Loki logging
- **System Metrics Collection**: Automatic collection of system metrics (CPU, memory, disk, GPU, etc.)
- **Alerting**: Configurable alerting based on metric thresholds
- **Frontend Integration**: HTTP endpoints for metrics that can be consumed by frontend dashboards
- **Compatibility Layer**: Backward compatibility with existing code

## Installation

The monitoring module is included as part of the NextGen FinGPT project. To install the required dependencies:

```bash
pip install prometheus_client psutil python-dotenv logging-loki
```

For GPU monitoring (optional):

```bash
pip install gputil py3nvml pynvml
```

## Usage

### Basic Usage

```python
from monitoring import setup_monitoring

# Set up monitoring for a service
monitor, metrics = setup_monitoring(
    service_name="my-service",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"environment": "production"}
)

# Log messages
monitor.log_info("Service started", component="main")
monitor.log_warning("Resource usage high", component="resource_monitor", usage=85)

# Update metrics
monitor.increment_counter("requests_total", 1, method="GET", endpoint="/api/data", status="200")
monitor.set_gauge("active_connections", 42, server="primary")
monitor.observe_histogram("response_time_seconds", 0.2, method="GET", endpoint="/api/data")
```

### Frontend Integration

The monitoring module exposes metrics via HTTP endpoints that can be consumed by frontend dashboards:

1. **Prometheus Metrics Endpoint**: By default, metrics are exposed on port 8010 (configurable via environment variable `PROMETHEUS_METRICS_PORT`).

   ```
   http://your-server:8010/metrics
   ```

2. **Grafana Integration**: You can configure Grafana to use Prometheus as a data source and create dashboards to visualize the metrics.

3. **Loki Logs**: Logs are sent to Loki (configurable via environment variable `LOKI_URL`), which can also be visualized in Grafana.

### System Metrics Collector

The module includes a standalone system metrics collector that can be run as a service:

```bash
# Install as a systemd service
sudo ./monitoring/setup_system_metrics.sh

# Or run directly
python -m monitoring.system_metrics
```

## Configuration

Configuration is done via environment variables:

- `PROMETHEUS_METRICS_PORT`: Port to expose Prometheus metrics on (default: 8010)
- `LOKI_URL`: URL of the Loki server (default: http://localhost:3100)
- `LOKI_USERNAME`: Username for Loki authentication (optional)
- `LOKI_PASSWORD`: Password for Loki authentication (optional)
- `METRICS_INTERVAL`: Interval for collecting system metrics in seconds (default: 5)
- `METRICS_SERVICE_NAME`: Name of the system metrics service (default: system_metrics)
- `METRICS_ENABLE_LOKI`: Whether to enable Loki logging for system metrics (default: true)

## API Reference

### MonitoringManager

The main class for unified monitoring:

- `log_info(message, **labels)`: Log an info message
- `log_warning(message, **labels)`: Log a warning message
- `log_error(message, **labels)`: Log an error message
- `log_critical(message, **labels)`: Log a critical message
- `increment_counter(name, value=1, **labels)`: Increment a counter
- `set_gauge(name, value, **labels)`: Set a gauge value
- `observe_histogram(name, value, **labels)`: Observe a value in a histogram

### PrometheusManager

For direct Prometheus metrics management:

- `create_counter(name, description, labels=None)`: Create a counter
- `create_gauge(name, description, labels=None)`: Create a gauge
- `create_histogram(name, description, labels=None, buckets=None)`: Create a histogram
- `create_summary(name, description, labels=None)`: Create a summary

### LokiManager

For direct Loki logging:

- `info(message, **labels)`: Log an info message
- `warning(message, **labels)`: Log a warning message
- `error(message, **labels)`: Log an error message
- `critical(message, **labels)`: Log a critical message
- `debug(message, **labels)`: Log a debug message

## Contributing

Contributions to the monitoring module are welcome. Please follow the project's contribution guidelines.
