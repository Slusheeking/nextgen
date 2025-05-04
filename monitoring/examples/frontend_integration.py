 #!/usr/bin/env python3
"""
Frontend Integration Example

This example demonstrates how to use the monitoring module with a frontend application.
It sets up a simple Flask API that exposes metrics and logs events that can be
visualized in a frontend dashboard.
"""

import os
import time
import random
from flask import Flask, jsonify, request

# Import the monitoring module
from monitoring import setup_monitoring

# Create a Flask app
app = Flask(__name__)

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="frontend-api",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "api-server"}
)

# Initialize request counter if not already created
if 'requests_total' not in metrics:
    metrics['requests_total'] = monitor.create_counter(
        "requests_total",
        "Total count of requests",
        ["method", "endpoint", "status"]
    )

# Initialize response time histogram if not already created
if 'response_time_seconds' not in metrics:
    metrics['response_time_seconds'] = monitor.create_histogram(
        "response_time_seconds",
        "Response time in seconds",
        ["method", "endpoint"]
    )

# Initialize active requests gauge if not already created
if 'active_requests' not in metrics:
    metrics['active_requests'] = monitor.create_gauge(
        "active_requests",
        "Number of active requests",
        ["method", "endpoint"]
    )

@app.before_request
def before_request():
    """Record the start time of each request and increment active requests."""
    request.start_time = time.time()
    endpoint = request.endpoint or 'unknown'
    metrics['active_requests'].inc(method=request.method, endpoint=endpoint)

@app.after_request
def after_request(response):
    """Record metrics after each request."""
    endpoint = request.endpoint or 'unknown'
    
    # Decrement active requests
    metrics['active_requests'].dec(method=request.method, endpoint=endpoint)
    
    # Record response time
    if hasattr(request, 'start_time'):
        response_time = time.time() - request.start_time
        monitor.observe_histogram(
            "response_time_seconds", 
            response_time, 
            method=request.method, 
            endpoint=endpoint
        )
    
    # Increment request counter
    monitor.increment_counter(
        "requests_total", 
        1, 
        method=request.method, 
        endpoint=endpoint, 
        status=response.status_code
    )
    
    # Log the request
    monitor.log_info(
        f"Request: {request.method} {request.path} {response.status_code}",
        method=request.method,
        path=request.path,
        status=response.status_code,
        response_time=response_time if hasattr(request, 'start_time') else None
    )
    
    return response

@app.route('/api/data')
def get_data():
    """Example API endpoint that returns some data."""
    # Simulate some processing time
    time.sleep(random.uniform(0.05, 0.2))
    
    # Return some data
    return jsonify({
        "data": [
            {"id": 1, "name": "Item 1", "value": random.randint(1, 100)},
            {"id": 2, "name": "Item 2", "value": random.randint(1, 100)},
            {"id": 3, "name": "Item 3", "value": random.randint(1, 100)}
        ],
        "timestamp": time.time()
    })

@app.route('/api/metrics')
def get_metrics():
    """
    Example API endpoint that returns current metrics.
    
    Note: This is just for demonstration. In a real application,
    you would use the Prometheus /metrics endpoint directly.
    """
    # Get the current values of some metrics
    active_requests = metrics['active_requests']._value.get(('GET', '/api/metrics'), 0)
    
    return jsonify({
        "active_requests": active_requests,
        "note": "This is just a demonstration. In a real application, use the Prometheus /metrics endpoint."
    })

@app.route('/api/error')
def simulate_error():
    """Example API endpoint that simulates an error."""
    # Log an error
    monitor.log_error(
        "An error occurred while processing the request",
        method=request.method,
        path=request.path,
        error_type="SimulatedError"
    )
    
    # Increment error counter
    monitor.increment_counter(
        "errors_total", 
        1, 
        type="SimulatedError", 
        code="500"
    )
    
    # Return an error response
    return jsonify({
        "error": "Simulated error",
        "timestamp": time.time()
    }), 500

if __name__ == '__main__':
    # Create the examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Log application startup
    monitor.log_info("Frontend API server starting", port=5000)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
