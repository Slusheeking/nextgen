# Prometheus Monitoring System

This directory contains the Prometheus monitoring system components for the NextGen FinGPT project.

## Components

### 1. Prometheus Server

Prometheus is a monitoring system and time series database. It collects metrics from configured targets at given intervals, evaluates rule expressions, displays the results, and can trigger alerts when specified conditions are observed.

- **Status**: Running as a system service (`prometheus.service`)
- **Configuration**: Located at `/etc/prometheus/prometheus.yml`
- **Data Storage**: Located at `/var/lib/prometheus/`

### 2. System Metrics Collector

The System Metrics Collector is a Python service that collects system metrics (CPU, memory, disk, network, GPU) and exposes them through Prometheus. It also sends alerts to Loki when metrics exceed configured thresholds.

- **Status**: Available but not running as a service yet
- **Setup**: Run `sudo ./setup_system_metrics.sh` to install and start the service
- **Service File**: `system-metrics.service`

### 3. Loki

Loki is a log aggregation system designed to store and query logs from all your applications and infrastructure.

- **Status**: Running as a system service (`loki.service`)
- **Configuration**: Located at `/etc/loki/local-config.yaml`

## Setup Instructions

### Setting up the System Metrics Collector

1. Make sure Prometheus and Loki are running
2. Run the setup script:
   ```bash
   sudo ./setup_system_metrics.sh
   ```
3. Verify the service is running:
   ```bash
   systemctl status system-metrics.service
   ```

### Viewing Metrics and Logs

- **Prometheus UI**: http://localhost:9090
- **Loki Logs**: Can be viewed through Grafana if installed

## Troubleshooting

### Checking Service Status

```bash
# Check Prometheus status
systemctl status prometheus

# Check Loki status
systemctl status loki

# Check System Metrics Collector status
systemctl status system-metrics
```

### Viewing Service Logs

```bash
# View Prometheus logs
journalctl -u prometheus -f

# View Loki logs
journalctl -u loki -f

# View System Metrics Collector logs
journalctl -u system-metrics -f
```

## Additional Information

- The System Metrics Collector uses the `prometheus_client` Python library to expose metrics
- Alerts are configured in the `SystemMetricsCollector` class with default thresholds
- GPU metrics collection is attempted if GPU monitoring libraries are available
