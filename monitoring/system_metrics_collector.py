#!/usr/bin/env python3
"""
System Metrics Collector

This module provides a standalone service for collecting system metrics
(CPU, memory, disk, GPU, etc.) and exposing them through Prometheus.
It also sends alerts to Loki when metrics exceed configured thresholds.
"""

import os
import time
import logging
import signal
import sys
import psutil
from pathlib import Path
from threading import Thread

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_metrics_collector")

# Import the monitoring module
from monitoring.monitoring import SystemMetricsCollector

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
                self.collect_cpu_metrics()
                
                # Collect memory metrics
                self.collect_memory_metrics()
                
                # Collect disk metrics
                self.collect_disk_metrics()
                
                # Collect network metrics
                self.collect_network_metrics()
                
                # Collect GPU metrics if available
                if self.collector.has_gpu:
                    if hasattr(self.collector, 'collect_gpu_metrics_nvml') and hasattr(self.collector, 'NVML_AVAILABLE') and self.collector.NVML_AVAILABLE:
                        self.collect_gpu_metrics_nvml()
                    elif hasattr(self.collector, 'collect_gpu_metrics_gputil'):
                        self.collect_gpu_metrics_gputil()
                
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
    
    def collect_cpu_metrics(self):
        """Collect CPU-related metrics."""
        # CPU percentage (per CPU)
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        for i, percent in enumerate(cpu_percent):
            self.collector.prom.set_gauge("cpu_utilization_percent", percent, cpu=f"cpu{i}")
        
        # Overall CPU percentage
        total_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.collector.prom.set_gauge("cpu_utilization_percent", total_cpu_percent, cpu="total")
        
        # Check for alerts on total CPU usage
        if self.collector.enable_loki:
            self.collector.check_and_alert("cpu_utilization_percent", total_cpu_percent, {"cpu": "total"})
        
        # CPU frequency
        cpu_freq = psutil.cpu_freq(percpu=True)
        if cpu_freq:
            for i, freq in enumerate(cpu_freq):
                if freq and hasattr(freq, 'current'):
                    self.collector.prom.set_gauge("cpu_frequency_mhz", freq.current, cpu=f"cpu{i}")
        
        # Overall CPU frequency
        cpu_freq_overall = psutil.cpu_freq()
        if cpu_freq_overall and hasattr(cpu_freq_overall, 'current'):
            self.collector.prom.set_gauge("cpu_frequency_mhz", cpu_freq_overall.current, cpu="total")
        
        # CPU count
        self.collector.prom.set_gauge("cpu_count", psutil.cpu_count(logical=True))
        
        # Load average (1, 5, 15 minutes)
        load_avg = psutil.getloadavg()
        self.collector.prom.set_gauge("load_average", load_avg[0], period="1min")
        self.collector.prom.set_gauge("load_average", load_avg[1], period="5min")
        self.collector.prom.set_gauge("load_average", load_avg[2], period="15min")
        
        # CPU temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for sensor_name, entries in temps.items():
                    for i, entry in enumerate(entries):
                        temp_value = entry.current
                        self.collector.prom.set_gauge(
                            "cpu_temperature_celsius", 
                            temp_value, 
                            sensor=f"{sensor_name}_{i}"
                        )
                        
                        # Alert on high CPU temperatures
                        if self.collector.enable_loki:
                            self.collector.check_and_alert("cpu_temperature_celsius", temp_value, {"sensor": f"{sensor_name}_{i}"})
        except Exception as e:
            logger.debug(f"Could not collect CPU temperature: {e}")
    
    def collect_memory_metrics(self):
        """Collect memory-related metrics."""
        # Virtual memory
        virtual_mem = psutil.virtual_memory()
        self.collector.prom.set_gauge("memory_usage_bytes", virtual_mem.total, type="total")
        self.collector.prom.set_gauge("memory_usage_bytes", virtual_mem.used, type="used")
        self.collector.prom.set_gauge("memory_usage_bytes", virtual_mem.available, type="available")
        self.collector.prom.set_gauge("memory_usage_bytes", virtual_mem.free, type="free")
        
        mem_percent = virtual_mem.percent
        self.collector.prom.set_gauge("memory_usage_percent", mem_percent, type="virtual")
        
        # Check for alerts on memory usage
        if self.collector.enable_loki:
            self.collector.check_and_alert("memory_usage_percent", mem_percent, {"type": "virtual"})
        
        # Swap memory
        swap_mem = psutil.swap_memory()
        self.collector.prom.set_gauge("memory_usage_bytes", swap_mem.total, type="swap_total")
        self.collector.prom.set_gauge("memory_usage_bytes", swap_mem.used, type="swap_used")
        self.collector.prom.set_gauge("memory_usage_bytes", swap_mem.free, type="swap_free")
        
        swap_percent = swap_mem.percent
        self.collector.prom.set_gauge("memory_usage_percent", swap_percent, type="swap")
        
        # Check for alerts on swap usage
        if self.collector.enable_loki and swap_mem.total > 0:  # Only alert if swap is enabled
            self.collector.check_and_alert("memory_usage_percent", swap_percent, {"type": "swap"})
    
    def collect_disk_metrics(self):
        """Collect disk-related metrics."""
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                
                # Extract device name without path
                device_name = os.path.basename(partition.device) if partition.device else "unknown"
                
                self.collector.prom.set_gauge(
                    "disk_usage_bytes",
                    usage.total,
                    device=device_name,
                    mountpoint=partition.mountpoint,
                    type="total"
                )
                
                self.collector.prom.set_gauge(
                    "disk_usage_bytes",
                    usage.used,
                    device=device_name,
                    mountpoint=partition.mountpoint,
                    type="used"
                )
                
                self.collector.prom.set_gauge(
                    "disk_usage_bytes",
                    usage.free,
                    device=device_name,
                    mountpoint=partition.mountpoint,
                    type="free"
                )
                
                disk_percent = usage.percent
                self.collector.prom.set_gauge(
                    "disk_usage_percent",
                    disk_percent,
                    device=device_name,
                    mountpoint=partition.mountpoint
                )
                
                # Check for alerts on disk usage
                if self.collector.enable_loki:
                    self.collector.check_and_alert("disk_usage_percent", disk_percent, {
                        "device": device_name,
                        "mountpoint": partition.mountpoint
                    })
            except (PermissionError, FileNotFoundError) as e:
                logger.debug(f"Could not collect disk usage for {partition.mountpoint}: {e}")
        
        # Disk I/O
        disk_io_counters = psutil.disk_io_counters(perdisk=True)
        for disk_name, counters in disk_io_counters.items():
            self.collector.prom.set_gauge("disk_io_bytes", counters.read_bytes, device=disk_name, direction="read")
            self.collector.prom.set_gauge("disk_io_bytes", counters.write_bytes, device=disk_name, direction="write")
    
    def collect_network_metrics(self):
        """Collect network-related metrics."""
        # Network I/O
        net_io_counters = psutil.net_io_counters(pernic=True)
        for nic_name, counters in net_io_counters.items():
            self.collector.prom.set_gauge("network_io_bytes", counters.bytes_sent, interface=nic_name, direction="sent")
            self.collector.prom.set_gauge("network_io_bytes", counters.bytes_recv, interface=nic_name, direction="received")
        
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
                self.collector.prom.set_gauge("network_connections", count, type=type_name, status=status)
        except (PermissionError, psutil.AccessDenied) as e:
            logger.debug(f"Could not collect network connections: {e}")
    
    def _get_connection_type_name(self, type_code):
        """Convert connection type code to name."""
        import socket
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
            import py3nvml.py3nvml as nvml
            
            for i in range(self.collector.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU name
                name = nvml.nvmlDeviceGetName(handle)
                
                # GPU utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                mem_util = utilization.memory
                
                self.collector.prom.set_gauge("gpu_utilization_percent", gpu_util, gpu=f"{i}:{name}", type="compute")
                self.collector.prom.set_gauge("gpu_utilization_percent", mem_util, gpu=f"{i}:{name}", type="memory")
                
                # Alert on high GPU utilization
                if self.collector.enable_loki:
                    self.collector.check_and_alert("gpu_utilization_percent", gpu_util, {
                        "gpu": f"{i}:{name}",
                        "type": "compute"
                    })
                
                # GPU memory
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                self.collector.prom.set_gauge("gpu_memory_bytes", memory.total, gpu=f"{i}:{name}", type="total")
                self.collector.prom.set_gauge("gpu_memory_bytes", memory.used, gpu=f"{i}:{name}", type="used")
                self.collector.prom.set_gauge("gpu_memory_bytes", memory.free, gpu=f"{i}:{name}", type="free")
                
                memory_percent = (memory.used / memory.total) * 100
                self.collector.prom.set_gauge("gpu_memory_percent", memory_percent, gpu=f"{i}:{name}")
                
                # Alert on high GPU memory usage
                if self.collector.enable_loki:
                    self.collector.check_and_alert("gpu_memory_percent", memory_percent, {"gpu": f"{i}:{name}"})
                
                # GPU temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                self.collector.prom.set_gauge("gpu_temperature_celsius", temp, gpu=f"{i}:{name}")
                
                # Alert on high GPU temperature
                if self.collector.enable_loki:
                    self.collector.check_and_alert("gpu_temperature_celsius", temp, {"gpu": f"{i}:{name}"})
                
                # GPU power
                try:
                    power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # convert mW to W
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # convert mW to W
                    self.collector.prom.set_gauge("gpu_power_watts", power_usage, gpu=f"{i}:{name}", type="usage")
                    self.collector.prom.set_gauge("gpu_power_watts", power_limit, gpu=f"{i}:{name}", type="limit")
                except nvml.NVMLError:
                    logger.debug(f"Could not collect power info for GPU {i}")
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics with NVML: {e}")
    
    def collect_gpu_metrics_gputil(self):
        """Collect GPU metrics using GPUtil."""
        try:
            import GPUtil
            
            # Refresh GPU information
            self.collector.gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(self.collector.gpus):
                # GPU utilization
                gpu_util = gpu.load * 100
                self.collector.prom.set_gauge("gpu_utilization_percent", gpu_util, gpu=f"{i}:{gpu.name}", type="compute")
                
                # Alert on high GPU utilization
                if self.collector.enable_loki:
                    self.collector.check_and_alert("gpu_utilization_percent", gpu_util, {
                        "gpu": f"{i}:{gpu.name}",
                        "type": "compute"
                    })
                
                # GPU memory
                self.collector.prom.set_gauge("gpu_memory_bytes", gpu.memoryTotal * 1024 * 1024, gpu=f"{i}:{gpu.name}", type="total")
                self.collector.prom.set_gauge("gpu_memory_bytes", gpu.memoryUsed * 1024 * 1024, gpu=f"{i}:{gpu.name}", type="used")
                self.collector.prom.set_gauge("gpu_memory_bytes", (gpu.memoryTotal - gpu.memoryUsed) * 1024 * 1024, gpu=f"{i}:{gpu.name}", type="free")
                
                memory_percent = gpu.memoryUtil * 100
                self.collector.prom.set_gauge("gpu_memory_percent", memory_percent, gpu=f"{i}:{gpu.name}")
                
                # Alert on high GPU memory usage
                if self.collector.enable_loki:
                    self.collector.check_and_alert("gpu_memory_percent", memory_percent, {"gpu": f"{i}:{gpu.name}"})
                
                # GPU temperature
                temp = gpu.temperature
                self.collector.prom.set_gauge("gpu_temperature_celsius", temp, gpu=f"{i}:{gpu.name}")
                
                # Alert on high GPU temperature
                if self.collector.enable_loki:
                    self.collector.check_and_alert("gpu_temperature_celsius", temp, {"gpu": f"{i}:{gpu.name}"})
        except Exception as e:
            logger.error(f"Error collecting GPU metrics with GPUtil: {e}")
    
    def stop(self):
        """Stop the service."""
        logger.info("Stopping System Metrics Collector service...")
        self.running = False


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
