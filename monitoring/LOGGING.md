# NextGen Logging System

This document describes the NextGen logging system, which has been updated to use file-based logging with a FastAPI server for front-end access.

## Overview

The logging system consists of the following components:

1. **Python Logging**: Standard Python logging is used to write logs to files.
2. **Log Files**: Logs are stored in the `logs` directory, with a separate file for each service and a master log file.
3. **Log API**: A FastAPI server provides access to the logs for the front end.

## Log Files

Logs are stored in the `logs` directory in the project root. The following log files are created:

- `master.log`: Contains all log messages from all services.
- `{service_name}.log`: Contains log messages from a specific service.

## Log API

The Log API is a FastAPI server that provides access to the log files. It offers the following endpoints:

- `GET /`: Returns information about the API.
- `GET /logs`: Returns a list of available log files.
- `GET /logs/{filename}`: Returns the content of a specific log file.
- `GET /logs/{filename}/stream`: Streams updates to a log file using Server-Sent Events (SSE).
- `GET /logs/master`: Returns the content of the master log file.
- `GET /logs/master/stream`: Streams updates to the master log file.
- `POST /logs`: Adds a log entry to the master log file.

## Usage

### Running the Log API Server

To run the Log API server, use the following command:

```bash
python monitoring/run_log_api.py
```

The server will be available at `http://localhost:8011`.

### Accessing Logs from the Front End

You can access the logs from the front end using the Log API. Here are some examples:

#### Get a List of Log Files

```javascript
fetch('http://localhost:8011/logs')
  .then(response => response.json())
  .then(data => console.log(data));
```

#### Get the Content of a Log File

```javascript
fetch('http://localhost:8011/logs/master.log')
  .then(response => response.json())
  .then(data => console.log(data.content));
```

#### Stream Updates to a Log File

```javascript
const eventSource = new EventSource('http://localhost:8011/logs/master.log/stream');
eventSource.onmessage = event => {
  const data = JSON.parse(event.data);
  console.log(data.line);
};
```

### Fixing Loki Endpoint Issues

If you encounter Loki endpoint issues, you can run the following script to remove Loki handlers and ensure all logs are properly written to files:

```bash
python monitoring/fix_loki_endpoint.py
```

## Logging in Your Code

To log messages in your code, use the `MonitoringManager` class from `monitoring.system_monitor`:

```python
from monitoring.system_monitor import MonitoringManager

# Create a monitoring manager
monitor = MonitoringManager(service_name="my-service")

# Log messages
monitor.log_info("This is an info message", component="my-component", action="my-action")
monitor.log_warning("This is a warning message", component="my-component", action="my-action")
monitor.log_error("This is an error message", component="my-component", action="my-action")
monitor.log_critical("This is a critical message", component="my-component", action="my-action")
```

The `MonitoringManager` class automatically writes logs to the appropriate log files and includes them in the master log file.

## Monitoring Metrics

In addition to logging, the `MonitoringManager` class also provides methods for collecting and exposing Prometheus metrics:

```python
# Increment a counter
monitor.increment_counter("requests_total", method="GET", endpoint="/api/data", status="200")

# Set a gauge
monitor.set_gauge("cpu_percent", 50.0)

# Observe a histogram
monitor.observe_histogram("request_duration_seconds", 0.1, method="GET", endpoint="/api/data")

# Observe a summary
monitor.observe_summary("operation_duration_seconds", 0.2, component="my-component", operation="my-operation")
```

Prometheus metrics are exposed on port 8010 by default.