# NextGen Monitoring Module

This module provides a simplified, unified monitoring solution for the NextGen FinGPT project, supporting Prometheus metrics and Loki logging with minimal setup.

## Features

- Unified MonitoringManager for Prometheus metrics and Loki logging
- Basic SystemMetricsCollector for CPU, memory, and disk metrics
- Minimal, robust API for easy integration

## Installation

Install the required dependencies:

```bash
pip install prometheus_client python-logging-loki psutil
```

## Usage

### Basic Example

```python
from monitoring.system_monitor import MonitoringManager, SystemMetricsCollector

# Initialize monitoring
monitor = MonitoringManager(service_name="my-service")

# Log messages
monitor.log_info("Service started")
monitor.log_warning("Resource usage high", usage=85)

# Update metrics
monitor.increment_counter("requests_total", method="GET", endpoint="/api/data", status="200")

# Start system metrics collection (optional)
collector = SystemMetricsCollector(monitor)
collector.start()
```

### Exposed Metrics

Prometheus metrics are exposed on port 8010 by default:

```
http://localhost:8010/
```

You can configure the port via the `metrics_port` argument to `MonitoringManager`.

### Loki Logging

Logs are sent to Loki at `http://localhost:3100/loki/api/v1/push` by default. You can configure the URL via the `loki_url` argument to `MonitoringManager`.

## API Reference

### MonitoringManager

- `log_info(message, **labels)`: Log an info message
- `log_warning(message, **labels)`: Log a warning message
- `log_error(message, **labels)`: Log an error message
- `log_critical(message, **labels)`: Log a critical message
- `log_debug(message, **labels)`: Log a debug message
- `increment_counter(name, value=1, **labels)`: Increment a counter metric
- `set_gauge(name, value, **labels)`: Set a gauge metric

### SystemMetricsCollector

- `start()`: Start collecting system metrics (CPU, memory, disk)
- `stop()`: Stop collecting system metrics

## Configuration

You can configure the monitoring system via arguments to `MonitoringManager` or environment variables:

- `metrics_port`: Port to expose Prometheus metrics (default: 8010)
- `loki_url`: URL for Loki logging (default: http://localhost:3100/loki/api/v1/push)

## Notes

- Only core system metrics (CPU, memory, disk) are collected by default.
- For additional metrics or logging customization, extend `system_monitor.py` as needed.

## Contributing

Contributions are welcome. Please follow the project's contribution guidelines.
