"""
Utility methods for monitoring system
"""

import os
from prometheus_client import Counter, Gauge, Histogram, Summary

class MonitoringUtils:
    """
    Utility methods for creating Prometheus metrics
    """
    
    @staticmethod
    def create_gauge(name, description, service_name):
        """Create a gauge metric"""
        try:
            metric = Gauge(
                f"{service_name}_{name}",
                description
            )
            return metric
        except Exception as e:
            print(f"Failed to create gauge {name}: {str(e)}")
            return None
            
    @staticmethod
    def create_counter(name, description, service_name):
        """Create a counter metric"""
        try:
            metric = Counter(
                f"{service_name}_{name}",
                description
            )
            return metric
        except Exception as e:
            print(f"Failed to create counter {name}: {str(e)}")
            return None
            
    @staticmethod
    def create_histogram(name, description, service_name, buckets=None):
        """Create a histogram metric"""
        try:
            metric = Histogram(
                f"{service_name}_{name}",
                description,
                buckets=buckets
            )
            return metric
        except Exception as e:
            print(f"Failed to create histogram {name}: {str(e)}")
            return None
            
    @staticmethod
    def create_summary(name, description, service_name):
        """Create a summary metric"""
        try:
            metric = Summary(
                f"{service_name}_{name}",
                description
            )
            return metric
        except Exception as e:
            print(f"Failed to create summary {name}: {str(e)}")
            return None