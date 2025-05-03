import os
import time
import threading
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client import start_http_server, push_to_gateway
from dotenv import load_dotenv

class PrometheusManager:
    """
    A manager class for Prometheus metrics functionality.
    Provides an interface for creating and managing Prometheus metrics,
    as well as exposing them via HTTP server or pushing to a Pushgateway.
    """
    
    def __init__(self, service_name=None, expose_metrics=True, metrics_port=None, 
                 pushgateway_url=None, pushgateway_job=None):
        """
        Initialize the Prometheus metrics manager.
        
        Args:
            service_name (str, optional): Name of the service for metrics identification.
                                          Defaults to env var or 'nextgen'.
            expose_metrics (bool): Whether to expose metrics via HTTP server. Defaults to True.
            metrics_port (int, optional): Port to expose metrics on. Defaults to env var or 8010.
            pushgateway_url (str, optional): URL of Prometheus Pushgateway if using push model.
                                            Defaults to env var or None.
            pushgateway_job (str, optional): Job name for Pushgateway. Defaults to service_name.
        """
        # Load environment variables
        load_dotenv()
        
        # Set service name from parameters or environment variables
        environment = os.getenv('ENVIRONMENT', 'development')
        service_version = os.getenv('SERVICE_VERSION', '1.0.0')
        default_name = f"nextgen-{environment}"
        self.service_name = service_name or os.getenv('SERVICE_NAME', default_name)
        
        # Set metrics configuration
        self.metrics_port = int(metrics_port or os.getenv('PROMETHEUS_METRICS_PORT', '8010'))
        self.pushgateway_url = pushgateway_url or os.getenv('PUSHGATEWAY_URL', None)
        self.pushgateway_job = pushgateway_job or self.service_name
        
        # Store environment info
        self.environment = environment
        self.version = service_version
        
        # Dictionary to store all metrics
        self.metrics = {}
        
        # Start metrics server if requested
        if expose_metrics:
            self._start_metrics_server()
        
        # Set up automatic push to gateway if URL provided
        self.push_thread = None
        if self.pushgateway_url:
            self.push_interval = int(os.getenv('PUSHGATEWAY_INTERVAL', '15'))
            self._start_push_thread()
    
    def _start_metrics_server(self):
        """Start the HTTP server to expose metrics to Prometheus."""
        start_http_server(self.metrics_port)
        print(f"Prometheus metrics exposed on port {self.metrics_port}")
    
    def _start_push_thread(self):
        """Start background thread to periodically push metrics to Pushgateway."""
        if self.push_thread is not None and self.push_thread.is_alive():
            return  # Thread already running
        
        def push_metrics():
            while True:
                try:
                    push_to_gateway(
                        self.pushgateway_url, 
                        job=self.pushgateway_job,
                        registry=None  # Use the default registry
                    )
                except Exception as e:
                    print(f"Error pushing to Pushgateway: {e}")
                
                time.sleep(self.push_interval)
        
        self.push_thread = threading.Thread(target=push_metrics, daemon=True)
        self.push_thread.start()
    
    def create_counter(self, name, description, labels=None):
        """
        Create a Prometheus Counter metric.
        
        Args:
            name (str): Name of the counter.
            description (str): Description of what the counter measures.
            labels (list, optional): List of label names for this counter.
        
        Returns:
            Counter: A Prometheus Counter object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        counter = Counter(metric_name, description, all_labels)
        self.metrics[name] = counter
        return counter
    
    def create_gauge(self, name, description, labels=None):
        """
        Create a Prometheus Gauge metric.
        
        Args:
            name (str): Name of the gauge.
            description (str): Description of what the gauge measures.
            labels (list, optional): List of label names for this gauge.
        
        Returns:
            Gauge: A Prometheus Gauge object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        gauge = Gauge(metric_name, description, all_labels)
        self.metrics[name] = gauge
        return gauge
    
    def create_histogram(self, name, description, labels=None, buckets=None):
        """
        Create a Prometheus Histogram metric.
        
        Args:
            name (str): Name of the histogram.
            description (str): Description of what the histogram measures.
            labels (list, optional): List of label names for this histogram.
            buckets (list, optional): Custom buckets for the histogram.
        
        Returns:
            Histogram: A Prometheus Histogram object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        kwargs = {}
        if buckets:
            kwargs['buckets'] = buckets
        
        histogram = Histogram(metric_name, description, all_labels, **kwargs)
        self.metrics[name] = histogram
        return histogram
    
    def create_summary(self, name, description, labels=None):
        """
        Create a Prometheus Summary metric.
        
        Args:
            name (str): Name of the summary.
            description (str): Description of what the summary measures.
            labels (list, optional): List of label names for this summary.
        
        Returns:
            Summary: A Prometheus Summary object.
        """
        # Create a name with service prefix
        metric_name = f"{self.service_name}_{name}"
        # Add default labels for environment and version
        default_labels = ['environment', 'version']
        all_labels = default_labels + (labels or [])
        
        summary = Summary(metric_name, description, all_labels)
        self.metrics[name] = summary
        return summary
    
    def get_metric(self, name):
        """
        Get a previously created metric by name.
        
        Args:
            name (str): Name of the metric to retrieve.
        
        Returns:
            The metric object if found, None otherwise.
        """
        return self.metrics.get(name)
    
    def increment_counter(self, name, value=1, **labels):
        """
        Increment a counter by a given value.
        
        Args:
            name (str): Name of the counter to increment.
            value (float): Value to increment by. Defaults to 1.
            **labels: Labels to apply to this counter increment.
        
        Returns:
            bool: True if successful, False if counter not found.
        """
        counter = self.get_metric(name)
        if not counter or not hasattr(counter, 'inc'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        counter.labels(**label_values).inc(value)
        return True
    
    def set_gauge(self, name, value, **labels):
        """
        Set a gauge to a given value.
        
        Args:
            name (str): Name of the gauge to set.
            value (float): Value to set.
            **labels: Labels to apply to this gauge.
        
        Returns:
            bool: True if successful, False if gauge not found.
        """
        gauge = self.get_metric(name)
        if not gauge or not hasattr(gauge, 'set'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        gauge.labels(**label_values).set(value)
        return True
    
    def observe_histogram(self, name, value, **labels):
        """
        Observe a value in a histogram.
        
        Args:
            name (str): Name of the histogram for observation.
            value (float): Value to observe.
            **labels: Labels to apply to this observation.
        
        Returns:
            bool: True if successful, False if histogram not found.
        """
        histogram = self.get_metric(name)
        if not histogram or not hasattr(histogram, 'observe'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        histogram.labels(**label_values).observe(value)
        return True
    
    def observe_summary(self, name, value, **labels):
        """
        Observe a value in a summary.
        
        Args:
            name (str): Name of the summary for observation.
            value (float): Value to observe.
            **labels: Labels to apply to this observation.
        
        Returns:
            bool: True if successful, False if summary not found.
        """
        summary = self.get_metric(name)
        if not summary or not hasattr(summary, 'observe'):
            return False
        
        # Add environment and version to labels
        label_values = {
            'environment': self.environment,
            'version': self.version,
            **labels
        }
        summary.labels(**label_values).observe(value)
        return True
    
    def force_push(self):
        """
        Force a push to the Pushgateway immediately.
        
        Returns:
            bool: True if successful, False if no Pushgateway URL configured.
        """
        if not self.pushgateway_url:
            return False
        
        try:
            push_to_gateway(
                self.pushgateway_url, 
                job=self.pushgateway_job,
                registry=None  # Use the default registry
            )
            return True
        except Exception as e:
            print(f"Error pushing to Pushgateway: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create a Prometheus manager
    prom_manager = PrometheusManager(service_name="nextgen-test", metrics_port=9091)
    
    # Create different types of metrics
    requests_counter = prom_manager.create_counter(
        "http_requests_total", 
        "Total count of HTTP requests",
        ["method", "endpoint", "status"]
    )
    
    response_time = prom_manager.create_histogram(
        "http_response_time_seconds",
        "HTTP response time in seconds",
        ["method", "endpoint"]
    )
    
    memory_gauge = prom_manager.create_gauge(
        "memory_usage_bytes",
        "Memory usage in bytes"
    )
    
    # Simulate metric updates
    for _ in range(10):
        # Increment request counter
        prom_manager.increment_counter(
            "http_requests_total", 
            1, 
            method="GET", 
            endpoint="/api/data", 
            status="200"
        )
        
        # Observe response time
        prom_manager.observe_histogram(
            "http_response_time_seconds",
            0.2 + (0.1 * ((_ % 5) / 5)),
            method="GET",
            endpoint="/api/data"
        )
        
        # Update memory gauge
        prom_manager.set_gauge("memory_usage_bytes", 100 * (1 + (_ % 10)))
        
        # Wait a bit
        time.sleep(1)
    
    print("Metrics generated. If server is running, visit http://localhost:9091 to see metrics")
    
    # Keep the program running to serve metrics
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down")
