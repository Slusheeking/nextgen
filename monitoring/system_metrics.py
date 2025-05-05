"""
System Metrics Collector for NextGen

This module provides enhanced system metrics collection, including:
- CPU usage (overall and per-core)
- Memory usage (RAM, swap)
- Disk usage and I/O
- Network traffic
- GPU metrics (if available)
- Process metrics

The metrics are collected and sent to Netdata for visualization.
"""

import time
import threading
import psutil
import platform
import socket

# Try to import GPU-specific libraries
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    from py3nvml import py3nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class SystemMetricsCollector:
    """
    Enhanced system metrics collector that sends metrics to Netdata
    and provides detailed hardware information.
    """
    
    def __init__(self, logger, interval=5):
        """
        Initialize the system metrics collector
        
        Args:
            logger: NetdataLogger instance for logging and metrics
            interval: Collection interval in seconds
        """
        self.logger = logger
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Initialize GPU monitoring if available
        self.has_gpu = False
        if HAS_NVML:
            try:
                py3nvml.nvmlInit()
                self.gpu_count = py3nvml.nvmlDeviceGetCount()
                self.has_gpu = self.gpu_count > 0
                if self.has_gpu:
                    self.logger.info(f"NVIDIA GPU monitoring enabled ({self.gpu_count} GPUs detected)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
        elif HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_count = len(gpus)
                self.has_gpu = self.gpu_count > 0
                if self.has_gpu:
                    self.logger.info(f"GPU monitoring enabled via GPUtil ({self.gpu_count} GPUs detected)")
            except Exception as e:
                self.logger.warning(f"Failed to get GPUs via GPUtil: {e}")
        
        # Get system information
        self.system_info = self._get_system_info()
        self.logger.info("System metrics collector initialized", 
                       system=self.system_info['system'],
                       cpu=self.system_info['cpu_model'],
                       cores=self.system_info['cpu_cores'],
                       memory_gb=self.system_info['memory_gb'])
    
    def _get_system_info(self):
        """Get detailed system information"""
        info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_model': 'Unknown',
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disks': [],
            'network_interfaces': [],
            'gpus': []
        }
        
        # Get CPU model
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            info['cpu_model'] = line.split(':', 1)[1].strip()
                            break
            except Exception:
                pass
        
        # Get disk information
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                info['disks'].append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'percent': usage.percent
                })
            except Exception:
                # Some mountpoints may not be accessible
                pass
        
        # Get network interfaces
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    info['network_interfaces'].append({
                        'interface': iface,
                        'address': addr.address,
                        'netmask': addr.netmask
                    })
        
        # Get GPU information
        if self.has_gpu:
            if HAS_NVML:
                try:
                    for i in range(self.gpu_count):
                        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
                        name = py3nvml.nvmlDeviceGetName(handle)
                        memory = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                        info['gpus'].append({
                            'index': i,
                            'name': name,
                            'memory_total_mb': round(memory.total / (1024**2), 2),
                            'memory_free_mb': round(memory.free / (1024**2), 2),
                            'memory_used_mb': round(memory.used / (1024**2), 2)
                        })
                except Exception as e:
                    self.logger.warning(f"Error getting GPU info via NVML: {e}")
            elif HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        info['gpus'].append({
                            'index': i,
                            'name': gpu.name,
                            'memory_total_mb': gpu.memoryTotal,
                            'memory_free_mb': gpu.memoryFree,
                            'memory_used_mb': gpu.memoryUsed,
                            'load': gpu.load,
                            'temperature': gpu.temperature
                        })
                except Exception as e:
                    self.logger.warning(f"Error getting GPU info via GPUtil: {e}")
        
        return info
    
    def start(self):
        """Start collecting system metrics"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_loop, daemon=True)
            self.thread.start()
            self.logger.info("System metrics collector started", interval=self.interval)
    
    def stop(self):
        """Stop collecting system metrics"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            self.logger.info("System metrics collector stopped")
            
            # Shutdown GPU monitoring if active
            if HAS_NVML and self.has_gpu:
                try:
                    py3nvml.nvmlShutdown()
                except Exception:
                    pass
    
    def _collect_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect and send metrics
                self._collect_cpu_metrics()
                self._collect_memory_metrics()
                self._collect_disk_metrics()
                self._collect_network_metrics()
                
                if self.has_gpu:
                    self._collect_gpu_metrics()
                
                # Sleep until next collection
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.interval)  # Sleep even on error
    
    def _collect_cpu_metrics(self):
        """Collect and send CPU metrics"""
        # Overall CPU usage
        cpu_percent = psutil.cpu_percent()
        self.logger.gauge("cpu.total_percent", cpu_percent)
        
        # Per-core CPU usage
        per_cpu = psutil.cpu_percent(percpu=True)
        for i, cpu in enumerate(per_cpu):
            self.logger.gauge(f"cpu.core{i}_percent", cpu)
        
        # CPU frequency
        try:
            freq = psutil.cpu_freq()
            if freq:
                self.logger.gauge("cpu.frequency_mhz", freq.current)
        except Exception:
            pass
        
        # CPU temperature (Linux only)
        if platform.system() == "Linux":
            try:
                temps = psutil.sensors_temperatures()
                if temps and 'coretemp' in temps:
                    for i, entry in enumerate(temps['coretemp']):
                        if hasattr(entry, 'current'):
                            self.logger.gauge(f"cpu.temp{i}_celsius", entry.current)
            except Exception:
                pass
        
        # CPU load averages (Unix-like systems)
        try:
            load1, load5, load15 = psutil.getloadavg()
            self.logger.gauge("system.load1", load1)
            self.logger.gauge("system.load5", load5)
            self.logger.gauge("system.load15", load15)
        except Exception:
            pass
    
    def _collect_memory_metrics(self):
        """Collect and send memory metrics"""
        # RAM usage
        mem = psutil.virtual_memory()
        self.logger.gauge("memory.total_bytes", mem.total)
        self.logger.gauge("memory.available_bytes", mem.available)
        self.logger.gauge("memory.used_bytes", mem.used)
        self.logger.gauge("memory.percent", mem.percent)
        
        # Swap usage
        swap = psutil.swap_memory()
        self.logger.gauge("memory.swap_total_bytes", swap.total)
        self.logger.gauge("memory.swap_used_bytes", swap.used)
        self.logger.gauge("memory.swap_percent", swap.percent)
    
    def _collect_disk_metrics(self):
        """Collect and send disk metrics"""
        # Disk usage
        for disk in self.system_info['disks']:
            try:
                usage = psutil.disk_usage(disk['mountpoint'])
                safe_name = disk['mountpoint'].replace('/', '_').replace('\\', '_')
                if safe_name == '':
                    safe_name = 'root'
                
                self.logger.gauge(f"disk.{safe_name}_total_bytes", usage.total)
                self.logger.gauge(f"disk.{safe_name}_used_bytes", usage.used)
                self.logger.gauge(f"disk.{safe_name}_percent", usage.percent)
            except Exception:
                pass
        
        # Disk I/O
        try:
            io_counters = psutil.disk_io_counters()
            self.logger.gauge("disk.read_bytes", io_counters.read_bytes)
            self.logger.gauge("disk.write_bytes", io_counters.write_bytes)
            self.logger.gauge("disk.read_count", io_counters.read_count)
            self.logger.gauge("disk.write_count", io_counters.write_count)
        except Exception:
            pass
    
    def _collect_network_metrics(self):
        """Collect and send network metrics"""
        try:
            net_io = psutil.net_io_counters()
            self.logger.gauge("network.bytes_sent", net_io.bytes_sent)
            self.logger.gauge("network.bytes_recv", net_io.bytes_recv)
            self.logger.gauge("network.packets_sent", net_io.packets_sent)
            self.logger.gauge("network.packets_recv", net_io.packets_recv)
            
            # Per-interface metrics
            net_io_per_nic = psutil.net_io_counters(pernic=True)
            for nic, counters in net_io_per_nic.items():
                safe_name = nic.replace('.', '_').replace('-', '_')
                self.logger.gauge(f"network.{safe_name}_bytes_sent", counters.bytes_sent)
                self.logger.gauge(f"network.{safe_name}_bytes_recv", counters.bytes_recv)
        except Exception:
            pass
    
    def _collect_gpu_metrics(self):
        """Collect and send GPU metrics if available"""
        if HAS_NVML:
            try:
                for i in range(self.gpu_count):
                    handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Memory usage
                    memory = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                    self.logger.gauge(f"gpu.{i}.memory_total_bytes", memory.total)
                    self.logger.gauge(f"gpu.{i}.memory_used_bytes", memory.used)
                    self.logger.gauge(f"gpu.{i}.memory_percent", (memory.used / memory.total) * 100)
                    
                    # Utilization
                    utilization = py3nvml.nvmlDeviceGetUtilizationRates(handle)
                    self.logger.gauge(f"gpu.{i}.gpu_utilization", utilization.gpu)
                    self.logger.gauge(f"gpu.{i}.memory_utilization", utilization.memory)
                    
                    # Temperature
                    temp = py3nvml.nvmlDeviceGetTemperature(handle, py3nvml.NVML_TEMPERATURE_GPU)
                    self.logger.gauge(f"gpu.{i}.temperature_celsius", temp)
                    
                    # Power usage
                    try:
                        power = py3nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                        self.logger.gauge(f"gpu.{i}.power_watts", power)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"Error collecting NVIDIA GPU metrics: {e}")
        elif HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.logger.gauge(f"gpu.{i}.memory_total_mb", gpu.memoryTotal)
                    self.logger.gauge(f"gpu.{i}.memory_used_mb", gpu.memoryUsed)
                    self.logger.gauge(f"gpu.{i}.memory_percent", (gpu.memoryUsed / gpu.memoryTotal) * 100)
                    self.logger.gauge(f"gpu.{i}.load_percent", gpu.load * 100)
                    self.logger.gauge(f"gpu.{i}.temperature_celsius", gpu.temperature)
            except Exception as e:
                self.logger.warning(f"Error collecting GPU metrics via GPUtil: {e}")


# Example usage
if __name__ == "__main__":
    from monitoring.netdata_logger import NetdataLogger
    
    logger = NetdataLogger("system-metrics")
    collector = SystemMetricsCollector(logger)
    
    print("Starting system metrics collector...")
    collector.start()
    
    try:
        # Keep the script running
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping system metrics collector...")
        collector.stop()
        print("Done.")