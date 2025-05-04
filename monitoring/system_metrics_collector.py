#!/usr/bin/env python3
"""
System Metrics Collector for Prometheus

This script collects various system metrics (CPU, memory, disk, GPU, etc.)
and exposes them through Prometheus for monitoring.
"""

import os
import time
import platform
import subprocess
import logging
import socket
import signal
import sys
import psutil
from pathlib import Path
from threading import Thread

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        GPU_AVAILABLE = True
    except ImportError:
        GPU_AVAILABLE = False

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Import our monitoring utilities
from monitoring.monitoring import PrometheusManager, LokiManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_metrics")

class SystemMetricsCollector:
    """
    Collects system metrics and exposes them through Prometheus.
    Sends system alerts and notifications to Loki.
    """
    
    def __init__(self, interval=5, service_name="system_metrics", enable_loki=True):
        """
        Initialize the system metrics collector.
        
        Args:
            interval (int): Collection interval in seconds. Defaults to 5.
            service_name (str): Name of the service for Prometheus. Defaults to "system_metrics".
            enable_loki (bool): Whether to enable Loki logging for alerts. Defaults to True.
        """
        self.interval = interval
        self.running = False
        self.metrics_thread = None
        
        # Initialize Prometheus manager
        self.prom = PrometheusManager(service_name=service_name)
        
        # Initialize Loki manager for alerts
        self.enable_loki = enable_loki
        if enable_loki:
            self.loki = LokiManager(service_name=service_name)
            logger.info("Loki logging initialized for system alerts")
        
        # Initialize metrics
        self._init_metrics()
        
        # Configure alert thresholds with default values
        self._configure_alert_thresholds()
        
        # Initialize GPU monitoring if available
        self.has_gpu = False
        if GPU_AVAILABLE or NVML_AVAILABLE:
            try:
                if NVML_AVAILABLE:
                    nvml.nvmlInit()
                    self.gpu_count = nvml.nvmlDeviceGetCount()
                    self.has_gpu = self.gpu_count > 0
                    if self.has_gpu:
                        logger.info(f"NVML initialized. Found {self.gpu_count} GPU(s)")
                elif GPU_AVAILABLE:
                    if 'gputil' in globals():
                        self.gpus = gputil.getGPUs()
                        self.has_gpu = len(self.gpus) > 0
                        if self.has_gpu:
                            logger.info(f"GPUtil initialized. Found {len(self.gpus)} GPU(s)")
                    else:  # pynvml
                        import pynvml
                        pynvml.nvmlInit()
                        self.gpu_count = pynvml.nvmlDeviceGetCount()
                        self.has_gpu = self.gpu_count > 0
                        if self.has_gpu:
                            logger.info(f"PYNVML initialized. Found {self.gpu_count} GPU(s)")
            except Exception as e:
                logger.error(f"Error initializing GPU monitoring: {e}")
                self.has_gpu = False
        
        if not self.has_gpu:
            logger.warning("No GPU detected or GPU monitoring libraries unavailable")
    
    def _configure_alert_thresholds(self):
        """Configure default alert thresholds for system metrics."""
        self.alert_thresholds = {
            # CPU related thresholds
            'cpu_utilization_percent': {
                'warning': 80.0,  # 80% utilization
                'critical': 95.0,  # 95% utilization
                'duration': 3,     # Alert after 3 consecutive readings
                'counter': 0,
                'last_alert': 0,   # Timestamp of last alert
                'cooldown': 300    # 5 minutes between alerts
            },
            # Memory related thresholds
            'memory_usage_percent': {
                'warning': 80.0,
                'critical': 95.0,
                'duration': 3,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 300
            },
            # Disk related thresholds
            'disk_usage_percent': {
                'warning': 80.0,
                'critical': 95.0,
                'duration': 1,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 3600  # 1 hour between disk alerts
            },
            # GPU related thresholds
            'gpu_utilization_percent': {
                'warning': 85.0,
                'critical': 98.0,
                'duration': 5,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 300
            },
            'gpu_memory_percent': {
                'warning': 80.0,
                'critical': 95.0,
                'duration': 3,
                'counter': 0,
                'last_alert': 0,
                'cooldown': 300
            },
            'gpu_temperature_celsius': {
                'warning': 80.0,    # 80°C is getting hot
                'critical': 90.0,   # 90°C is too hot
                'duration': 1,      # Alert immediately on high temperature
                'counter': 0,
                'last_alert': 0,
                'cooldown': 60      # 1 minute between temperature alerts
            }
        }
        
        logger.info("Alert thresholds configured with default values")
    
    def set_alert_threshold(self, metric, warning=None, critical=None, duration=None, cooldown=None):
        """
        Set custom alert thresholds for a specific metric.
        
        Args:
            metric (str): Name of the metric (e.g., 'cpu_utilization_percent')
            warning (float, optional): Warning threshold value
            critical (float, optional): Critical threshold value
            duration (int, optional): Number of consecutive readings before alerting
            cooldown (int, optional): Minimum seconds between repeated alerts
        
        Returns:
            bool: True if threshold was set, False if metric not found
        """
        if metric not in self.alert_thresholds:
            logger.warning(f"Attempted to set threshold for unknown metric: {metric}")
            return False
            
        if warning is not None:
            self.alert_thresholds[metric]['warning'] = float(warning)
        if critical is not None:
            self.alert_thresholds[metric]['critical'] = float(critical)
        if duration is not None:
            self.alert_thresholds[metric]['duration'] = int(duration)
        if cooldown is not None:
            self.alert_thresholds[metric]['cooldown'] = int(cooldown)
            
        logger.info(f"Updated alert thresholds for {metric}: {self.alert_thresholds[metric]}")
        return True
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        # CPU metrics
        self.cpu_percent = self.prom.create_gauge(
            "cpu_utilization_percent", 
            "CPU utilization percentage",
            ["cpu"]
        )
        
        self.cpu_freq = self.prom.create_gauge(
            "cpu_frequency_mhz",
            "CPU frequency in MHz",
            ["cpu"]
        )
        
        self.cpu_count = self.prom.create_gauge(
            "cpu_count",
            "Number of CPU cores/threads"
        )
        
        self.load_avg = self.prom.create_gauge(
            "load_average",
            "System load average",
            ["period"]
        )
        
        self.cpu_temp = self.prom.create_gauge(
            "cpu_temperature_celsius",
            "CPU temperature in Celsius",
            ["sensor"]
        )
        
        # Memory metrics
        self.memory_usage = self.prom.create_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["type"]
        )
        
        self.memory_percent = self.prom.create_gauge(
            "memory_usage_percent",
            "Memory usage percentage",
            ["type"]
        )
        
        # Disk metrics
        self.disk_usage = self.prom.create_gauge(
            "disk_usage_bytes",
            "Disk usage in bytes",
            ["device", "mountpoint", "type"]
        )
        
        self.disk_percent = self.prom.create_gauge(
            "disk_usage_percent",
            "Disk usage percentage",
            ["device", "mountpoint"]
        )
        
        self.disk_io = self.prom.create_gauge(
            "disk_io_bytes",
            "Disk I/O in bytes",
            ["device", "direction"]
        )
        
        # Network metrics
        self.network_io = self.prom.create_gauge(
            "network_io_bytes",
            "Network I/O in bytes",
            ["interface", "direction"]
        )
        
        self.network_connections = self.prom.create_gauge(
            "network_connections",
            "Number of network connections",
            ["type", "status"]
        )
        
        # GPU metrics (if available)
        if GPU_AVAILABLE or NVML_AVAILABLE:
            self.gpu_utilization = self.prom.create_gauge(
                "gpu_utilization_percent",
                "GPU utilization percentage",
                ["gpu", "type"]
            )
            
            self.gpu_memory = self.prom.create_gauge(
                "gpu_memory_bytes",
                "GPU memory usage in bytes",
                ["gpu", "type"]
            )
            
            self.gpu_memory_percent = self.prom.create_gauge(
                "gpu_memory_percent",
                "GPU memory usage percentage",
                ["gpu"]
            )
            
            self.gpu_temp = self.prom.create_gauge(
                "gpu_temperature_celsius",
                "GPU temperature in Celsius",
                ["gpu"]
            )
            
            self.gpu_power = self.prom.create_gauge(
                "gpu_power_watts",
                "GPU power usage in watts",
                ["gpu", "type"]
            )
        
        # System uptime
        self.uptime = self.prom.create_gauge(
            "system_uptime_seconds",
            "System uptime in seconds"
        )
        
        # Process count
        self.process_count = self.prom.create_gauge(
            "process_count",
            "Number of running processes",
            ["state"]
        )
    
    def check_and_alert(self, metric_name, value, labels=None):
        """
        Check if a metric value exceeds alert thresholds and send alert if needed.
        
        Args:
            metric_name (str): Name of the metric
            value (float): Current value of the metric
            labels (dict, optional): Additional labels for this metric
        """
        if not self.enable_loki or metric_name not in self.alert_thresholds:
            return
            
        thresholds = self.alert_thresholds[metric_name]
        current_time = time.time()
        
        # Format labels for alert message
        label_str = ""
        if labels:
            label_str = " (" + ", ".join(f"{k}={v}" for k, v in labels.items()) + ")"
        
        # Check if value exceeds thresholds
        if value >= thresholds['critical']:
            thresholds['counter'] += 1
            
            if (thresholds['counter'] >= thresholds['duration'] and 
                current_time - thresholds['last_alert'] > thresholds['cooldown']):
                # Send CRITICAL alert to Loki
                alert_msg = f"CRITICAL: {metric_name}{label_str} = {value:.2f} exceeds critical threshold of {thresholds['critical']}"
                self.loki.critical(alert_msg, metric=metric_name, value=value, threshold="critical", **labels if labels else {})
                logger.critical(alert_msg)
                
                # Reset counter and update last alert time
                thresholds['counter'] = 0
                thresholds['last_alert'] = current_time
                
        elif value >= thresholds['warning']:
            thresholds['counter'] += 1
            
            if (thresholds['counter'] >= thresholds['duration'] and 
                current_time - thresholds['last_alert'] > thresholds['cooldown']):
                # Send WARNING alert to Loki
                alert_msg = f"WARNING: {metric_name}{label_str} = {value:.2f} exceeds warning threshold of {thresholds['warning']}"
                self.loki.warning(alert_msg, metric=metric_name, value=value, threshold="warning", **labels if labels else {})
                logger.warning(alert_msg)
                
                # Reset counter and update last alert time
                thresholds['counter'] = 0
                thresholds['last_alert'] = current_time
                
        else:
            # Reset counter if value returns to normal
            if thresholds['counter'] > 0:
                thresholds['counter'] = 0
                
                # Log recovery if we were previously above thresholds
                if current_time - thresholds['last_alert'] < thresholds['cooldown']:
                    recovery_msg = f"RECOVERY: {metric_name}{label_str} = {value:.2f} returned to normal"
                    if self.enable_loki:
                        self.loki.info(recovery_msg, metric=metric_name, value=value, status="recovery", **labels if labels else {})
                    logger.info(recovery_msg)
    
    def collect_cpu_metrics(self):
        """Collect CPU-related metrics."""
        # CPU percentage (per CPU)
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        for i, percent in enumerate(cpu_percent):
            self.prom.set_gauge("cpu_utilization_percent", percent, cpu=f"cpu{i}")
        
        # Overall CPU percentage
        total_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.prom.set_gauge("cpu_utilization_percent", total_cpu_percent, cpu="total")
        
        # Check for alerts on total CPU usage
        if self.enable_loki:
            self.check_and_alert("cpu_utilization_percent", total_cpu_percent, {"cpu": "total"})
        
        # CPU frequency
        cpu_freq = psutil.cpu_freq(percpu=True)
        if cpu_freq:
            for i, freq in enumerate(cpu_freq):
                if freq and hasattr(freq, 'current'):
                    self.prom.set_gauge("cpu_frequency_mhz", freq.current, cpu=f"cpu{i}")
        
        # Overall CPU frequency
        cpu_freq_overall = psutil.cpu_freq()
        if cpu_freq_overall and hasattr(cpu_freq_overall, 'current'):
            self.prom.set_gauge("cpu_frequency_mhz", cpu_freq_overall.current, cpu="total")
        
        # CPU count
        self.prom.set_gauge("cpu_count", psutil.cpu_count(logical=True))
        
        # Load average (1, 5, 15 minutes)
        load_avg = psutil.getloadavg()
        self.prom.set_gauge("load_average", load_avg[0], period="1min")
        self.prom.set_gauge("load_average", load_avg[1], period="5min")
        self.prom.set_gauge("load_average", load_avg[2], period="15min")
        
        # CPU temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for sensor_name, entries in temps.items():
                    for i, entry in enumerate(entries):
                        temp_value = entry.current
                        self.prom.set_gauge(
                            "cpu_temperature_celsius", 
                            temp_value, 
                            sensor=f"{sensor_name}_{i}"
                        )
                        
                        # Alert on high CPU temperatures
                        if self.enable_loki:
                            self.check_and_alert("cpu_temperature_celsius", temp_value, {"sensor": f"{sensor_name}_{i}"})
        except Exception as e:
            logger.debug(f"Could not collect CPU temperature: {e}")
    
    def collect_memory_metrics(self):
        """Collect memory-related metrics."""
        # Virtual memory
        virtual_mem = psutil.virtual_memory()
        self.prom.set_gauge("memory_usage_bytes", virtual_mem.total, type="total")
        self.prom.set_gauge("memory_usage_bytes", virtual_mem.used, type="used")
        self.prom.set_gauge("memory_usage_bytes", virtual_mem.available, type="available")
        self.prom.set_gauge("memory_usage_bytes", virtual_mem.free, type="free")
        
        mem_percent = virtual_mem.percent
        self.prom.set_gauge("memory_usage_percent", mem_percent, type="virtual")
        
        # Check for alerts on memory usage
        if self.enable_loki:
            self.check_and_alert("memory_usage_percent", mem_percent, {"type": "virtual"})
        
        # Swap memory
        swap_mem = psutil.swap_memory()
        self.prom.set_gauge("memory_usage_bytes", swap_mem.total, type="swap_total")
        self.prom.set_gauge("memory_usage_bytes", swap_mem.used, type="swap_used")
        self.prom.set_gauge("memory_usage_bytes", swap_mem.free, type="swap_free")
        
        swap_percent = swap_mem.percent
        self.prom.set_gauge("memory_usage_percent", swap_percent, type="swap")
        
        # Check for alerts on swap usage
        if self.enable_loki and swap_mem.total > 0:  # Only alert if swap is enabled
            self.check_and_alert("memory_usage_percent", swap_percent, {"type": "swap"})
    
    def collect_disk_metrics(self):
        """Collect disk-related metrics."""
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                
                # Extract device name without path
                device_name = os.path.basename(partition.device) if partition.device else "unknown"
                
                self.prom.set_gauge(
                    "disk_usage_bytes",
                    usage.total,
                    device=device_name,
                    mountpoint=partition.mountpoint,
                    type="total"
                )
                
                self.prom.set_gauge(
                    "disk_usage_bytes",
                    usage.used,
                    device=device_name,
                    mountpoint=partition.mountpoint,
                    type="used"
                )
                
                self.prom.set_gauge(
                    "disk_usage_bytes",
                    usage.free,
                    device=device_name,
                    mountpoint=partition.mountpoint,
                    type="free"
                )
                
                disk_percent = usage.percent
                self.prom.set_gauge(
                    "disk_usage_percent",
                    disk_percent,
                    device=device_name,
                    mountpoint=partition.mountpoint
                )
                
                # Check for alerts on disk usage
                if self.enable_loki:
                    self.check_and_alert("disk_usage_percent", disk_percent, {
                        "device": device_name,
                        "mountpoint": partition.mountpoint
                    })
            except (PermissionError, FileNotFoundError) as e:
                logger.debug(f"Could not collect disk usage for {partition.mountpoint}: {e}")
        
        # Disk I/O
        disk_io_counters = psutil.disk_io_counters(perdisk=True)
        for disk_name, counters in disk_io_counters.items():
            self.prom.set_gauge("disk_io_bytes", counters.read_bytes, device=disk_name, direction="read")
            self.prom.set_gauge("disk_io_bytes", counters.write_bytes, device=disk_name, direction="write")
    
    def collect_network_metrics(self):
        """Collect network-related metrics."""
        # Network I/O
        net_io_counters = psutil.net_io_counters(pernic=True)
        for nic_name, counters in net_io_counters.items():
            self.prom.set_gauge("network_io_bytes", counters.bytes_sent, interface=nic_name, direction="sent")
            self.prom.set_gauge("network_io_bytes", counters.bytes_recv, interface=nic_name, direction="received")
        
        # Network connections
        try:
            connections = psutil.net_connections()
            conn_status = {}
            
            for conn in connections:
                conn_type = conn.type
                status = conn.status if hasattr(conn, 'status') else 'UNKNOWN'
                
                key = (conn_type, status)
                if key in conn_status:
                    conn_status[key] += 1
                else:
                    conn_status[key] = 1
            
            for (conn_type, status), count in conn_status.items():
                type_name = self._get_connection_type_name(conn_type)
                self.prom.set_gauge("network_connections", count, type=type_name, status=status)
        except (PermissionError, psutil.AccessDenied) as e:
            logger.debug(f"Could not collect network connections: {e}")
    
    def _get_connection_type_name(self, type_code):
        """Convert connection type code to name."""
        type_map = {
            socket.SOCK_STREAM: 'TCP',
            socket.SOCK_DGRAM: 'UDP',
            socket.SOCK_RAW: 'RAW',
            socket.SOCK_RDM: 'RDM',
            socket.SOCK_SEQPACKET: 'SEQPACKET'
        }
        return type_map.get(type_code, f'UNKNOWN({type_code})')
    
    def collect_gpu_metrics_nvml(self):
        """Collect GPU metrics using NVML."""
        try:
            for i in range(self.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU name
                name = nvml.nvmlDeviceGetName(handle)
                
                # GPU utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                mem_util = utilization.memory
                
                self.prom.set_gauge("gpu_utilization_percent", gpu_util, gpu=f"{i}:{name}", type="compute")
                self.prom.set_gauge("gpu_utilization_percent", mem_util, gpu=f"{i}:{name}", type="memory")
                
                # Alert on high GPU utilization
                if self.enable_loki:
                    self.check_and_alert("gpu_utilization_percent", gpu_util, {
                        "gpu": f"{i}:{name}",
                        "type": "compute"
                    })
                
                # GPU memory
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                self.prom.set_gauge("gpu_memory_bytes", memory.total, gpu=f"{i}:{name}", type="total")
                self.prom.set_gauge("gpu_memory_bytes", memory.used, gpu=f"{i}:{name}", type="used")
                self.prom.set_gauge("gpu_memory_bytes", memory.free, gpu=f"{i}:{name}", type="free")
                
                memory_percent = (memory.used / memory.total) * 100
                self.prom.set_gauge("gpu_memory_percent", memory_percent, gpu=f"{i}:{name}")
                
                # Alert on high GPU memory usage
                if self.enable_loki:
                    self.check_and_alert("gpu_memory_percent", memory_percent, {"gpu": f"{i}:{name}"})
                
                # GPU temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                self.prom.set_gauge("gpu_temperature_celsius", temp, gpu=f"{i}:{name}")
                
                # Alert on high GPU temperature
                if self.enable_loki:
                    self.check_and_alert("gpu_temperature_celsius", temp, {"gpu": f"{i}:{name}"})
                
                # GPU power
                try:
                    power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # convert mW to W
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # convert mW to W
                    self.prom.set_gauge("gpu_power_watts", power_usage, gpu=f"{i}:{name}", type="usage")
                    self.prom.set_gauge("gpu_power_watts", power_limit, gpu=f"{i}:{name}", type="limit")
                except nvml.NVMLError:
                    logger.debug(f"Could not collect power info for GPU {i}")
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics with NVML: {e}")
    
    def collect_gpu_metrics_gputil(self):
        """Collect GPU metrics using GPUtil."""
        try:
            if 'gputil' in globals():
                # Refresh GPU information
                self.gpus = gputil.getGPUs()
                
                for i, gpu in enumerate(self.gpus):
                    # GPU utilization
                    gpu_util = gpu.load * 100
                    self.prom.set_gauge("gpu_utilization_percent", gpu_util, gpu=f"{i}:{gpu.name}", type="compute")
                    
                    # Alert on high GPU utilization
                    if self.enable_loki:
                        self.check_and_alert("gpu_utilization_percent", gpu_util, {
                            "gpu": f"{i}:{gpu.name}",
                            "type": "compute"
                        })
                    
                    # GPU memory
                    self.prom.set_gauge("gpu_memory_bytes", gpu.memoryTotal * 1024 * 1024, gpu=f"{i}:{gpu.name}", type="total")
                    self.prom.set_gauge("gpu_memory_bytes", gpu.memoryUsed * 1024 * 1024, gpu=f"{i}:{gpu.name}", type="used")
                    self.prom.set_gauge("gpu_memory_bytes", (gpu.memoryTotal - gpu.memoryUsed) * 1024 * 1024, gpu=f"{i}:{gpu.name}", type="free")
                    
                    memory_percent = gpu.memoryUtil * 100
                    self.prom.set_gauge("gpu_memory_percent", memory_percent, gpu=f"{i}:{gpu.name}")
                    
                    # Alert on high GPU memory usage
                    if self.enable_loki:
                        self.check_and_alert("gpu_memory_percent", memory_percent, {"gpu": f"{i}:{gpu.name}"})
                    
                    # GPU temperature
                    temp = gpu.temperature
                    self.prom.set_gauge("gpu_temperature_celsius", temp, gpu=f"{i}:{gpu.name}")
                    
                    # Alert on high GPU temperature
                    if self.enable_loki:
                        self.check_and_alert("gpu_temperature_celsius", temp, {"gpu": f"{i}:{gpu.name}"})
            else:  # pynvml
                import pynvml
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU name
                    name = pynvml.nvmlDeviceGetName(handle)
                    
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    mem_util = utilization.memory
                    
                    self.prom.set_gauge("gpu_utilization_percent", gpu_util, gpu=f"{i}:{name}", type="compute")
                    self.prom.set_gauge("gpu_utilization_percent", mem_util, gpu=f"{i}:{name}", type="memory")
                    
                    # Alert on high GPU utilization
                    if self.enable_loki:
                        self.check_and_alert("gpu_utilization_percent", gpu_util, {
                            "gpu": f"{i}:{name}",
                            "type": "compute"
                        })
                    
                    # GPU memory
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.prom.set_gauge("gpu_memory_bytes", memory.total, gpu=f"{i}:{name}", type="total")
                    self.prom.set_gauge("gpu_memory_bytes", memory.used, gpu=f"{i}:{name}", type="used")
                    self.prom.set_gauge("gpu_memory_bytes", memory.free, gpu=f"{i}:{name}", type="free")
                    
                    memory_percent = (memory.used / memory.total) * 100
                    self.prom.set_gauge("gpu_memory_percent", memory_percent, gpu=f"{i}:{name}")
                    
                    # Alert on high GPU memory usage
                    if self.enable_loki:
                        self.check_and_alert("gpu_memory_percent", memory_percent, {"gpu": f"{i}:{name}"})
                    
                    # GPU temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.prom.set_gauge("gpu_temperature_celsius", temp, gpu=f"{i}:{name}")
                    
                    # Alert on high GPU temperature
                    if self.enable_loki:
                        self.check_and_alert("gpu_temperature_celsius", temp, {"gpu": f"{i}:{name}"})
        except Exception as e:
            logger.error(f"Error collecting GPU metrics with GPUtil/pynvml: {e}")


class SystemMetricsService:
    """
    Service wrapper for the SystemMetricsCollector.
    Handles initialization, running, and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize the service."""
        self.running = False
        self.collector = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def start(self):
        """Start the system metrics collector service."""
        logger.info("Starting System Metrics Collector service...")
        
        # Initialize the collector
        try:
            # Get configuration from environment variables
            interval = int(os.getenv('METRICS_INTERVAL', '5'))
            service_name = os.getenv('METRICS_SERVICE_NAME', 'system_metrics')
            enable_loki = os.getenv('METRICS_ENABLE_LOKI', 'true').lower() == 'true'
            
            # Create the collector
            self.collector = SystemMetricsCollector(
                interval=interval,
                service_name=service_name,
                enable_loki=enable_loki
            )
            
            logger.info(f"SystemMetricsCollector initialized with interval={interval}s")
            
            # Start collecting metrics
            self.running = True
            self.run_metrics_loop()
            
        except Exception as e:
            logger.error(f"Error starting System Metrics Collector: {e}")
            sys.exit(1)
    
    def run_metrics_loop(self):
        """Run the metrics collection loop."""
        logger.info("Starting metrics collection loop...")
        
        while self.running:
            try:
                # Collect CPU metrics
                self.collector.collect_cpu_metrics()
                
                # Collect memory metrics
                self.collector.collect_memory_metrics()
                
                # Collect disk metrics
                self.collector.collect_disk_metrics()
                
                # Collect network metrics
                self.collector.collect_network_metrics()
                
                # Collect GPU metrics if available
                if self.collector.has_gpu:
                    if hasattr(self.collector, 'collect_gpu_metrics_nvml') and hasattr(self.collector, 'NVML_AVAILABLE') and self.collector.NVML_AVAILABLE:
                        self.collector.collect_gpu_metrics_nvml()
                    elif hasattr(self.collector, 'collect_gpu_metrics_gputil'):
                        self.collector.collect_gpu_metrics_gputil()
                
                # Collect system uptime
                uptime = time.time() - psutil.boot_time()
                self.collector.prom.set_gauge("system_uptime_seconds", uptime)
                
                # Collect process count
                process_count = len(psutil.pids())
                self.collector.prom.set_gauge("process_count", process_count, state="total")
                
                # Sleep for the configured interval
                time.sleep(self.collector.interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                # Sleep for a short time to avoid tight error loops
                time.sleep(1)
    
    def stop(self):
        """Stop the service."""
        logger.info("Stopping System Metrics Collector service...")
        self.running = False


# Main entry point
if __name__ == "__main__":
    # Create and start the service
    service = SystemMetricsService()
    
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        service.stop()
        logger.info("System Metrics Collector service stopped")
