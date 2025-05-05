# Netdata Logger

A unified interface for Netdata metrics and Python logging in the NextGen platform.

## Overview

The `NetdataLogger` class provides a simple, unified way to:

1. Send custom metrics to Netdata for visualization
2. Log messages with structured data using Python's logging system
3. Automatically collect and monitor system resources

This integration allows you to monitor both system metrics and application-specific metrics in real-time through the Netdata dashboard, while maintaining comprehensive logging for debugging and analysis.

## Installation

The `NetdataLogger` is already integrated into the NextGen platform. Make sure you have:

1. Netdata installed and running (via `monitoring/setup_netdata.sh`)
2. Python logging configured (automatically handled by `NetdataLogger`)

## Basic Usage

```python
from monitoring.netdata_logger import NetdataLogger

# Create a logger for your component
logger = NetdataLogger(component_name="my-component")

# Log messages
logger.info("Component started")
logger.warning("Resource usage high", cpu=85.2, memory=75.8)
logger.error("Database connection failed", db_host="db.example.com")

# Send metrics to Netdata
logger.gauge("api_response_time", 0.235)
logger.counter("api_requests", 1)
logger.timing("query_time_ms", 56)

# Start system metrics collection
logger.start_system_metrics()

# ... your application code ...

# Stop system metrics collection when done
logger.stop_system_metrics()
```

## Logging Methods

The `NetdataLogger` provides the standard logging methods:

- `logger.debug(message, **kwargs)` - Debug-level messages
- `logger.info(message, **kwargs)` - Informational messages
- `logger.warning(message, **kwargs)` - Warning messages
- `logger.error(message, **kwargs)` - Error messages
- `logger.critical(message, **kwargs)` - Critical error messages

All methods accept additional keyword arguments that are included as structured data in the logs.

## Metrics Methods

The `NetdataLogger` provides methods to send different types of metrics to Netdata:

- `logger.gauge(name, value)` - Values that can go up and down (e.g., CPU usage)
- `logger.counter(name, value=1)` - Values that only increase (e.g., request count)
- `logger.timing(name, value)` - Duration measurements in milliseconds
- `logger.histogram(name, value)` - Distribution of values

## System Metrics Collection

The `NetdataLogger` can automatically collect and send system metrics to Netdata:

```python
# Start collecting system metrics every 5 seconds
logger.start_system_metrics(interval=5)

# Stop collecting system metrics
logger.stop_system_metrics()
```

Collected metrics include:
- CPU usage
- Memory usage
- Disk usage
- Network I/O
- Component uptime

## Integration Examples

### In a Class

```python
class ApiService:
    def __init__(self):
        self.logger = NetdataLogger(component_name="api-service")
        self.logger.start_system_metrics()
    
    def handle_request(self, request_id, endpoint):
        self.logger.info("Handling request", request_id=request_id, endpoint=endpoint)
        
        # Measure request duration
        start_time = time.time()
        result = self._process_request()
        duration = time.time() - start_time
        
        # Record metrics
        self.logger.timing("request_duration_ms", duration * 1000)
        self.logger.counter("requests_total")
        
        return result
```

### In a Data Processing Pipeline

```python
class DataProcessor:
    def __init__(self):
        self.logger = NetdataLogger(component_name="data-processor")
    
    def process_batch(self, batch_id, items):
        self.logger.info("Processing batch", batch_id=batch_id, batch_size=len(items))
        
        start_time = time.time()
        # Process items...
        duration = time.time() - start_time
        
        self.logger.gauge("batch_processing_time", duration)
        self.logger.counter("processed_items", len(items))
```

### In a Trading Model

```python
class TradingModel:
    def __init__(self, model_name):
        self.logger = NetdataLogger(component_name=f"trading-model-{model_name}")
    
    def predict(self, market_data):
        start_time = time.time()
        prediction = self._run_model(market_data)
        duration = time.time() - start_time
        
        self.logger.timing("prediction_time_ms", duration * 1000)
        self.logger.gauge("prediction_confidence", prediction.confidence)
```

## Viewing Logs and Metrics

### Logs

Logs are written to:
- Component-specific log file: `logs/{component_name}.log`
- Master log file: `logs/master.log`

You can view logs through:
- The log API: `http://localhost:8011/logs`
- The custom dashboard: `http://localhost:8080`

### Metrics

Metrics are sent to Netdata and can be viewed through:
- Netdata dashboard: `http://localhost:19999`
- Custom dashboard: `http://localhost:8080`

## Advanced Configuration

### Custom Log Directory

```python
logger = NetdataLogger(
    component_name="my-component",
    log_dir="/path/to/custom/logs"
)
```

### Custom StatsD Endpoint

```python
logger = NetdataLogger(
    component_name="my-component",
    statsd_host="metrics.example.com",
    statsd_port=8125
)
```

## Complete Example

See `monitoring/example_usage.py` for a complete example of how to use `NetdataLogger` in different scenarios.