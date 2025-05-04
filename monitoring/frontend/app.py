#!/usr/bin/env python3
"""
Monitoring Frontend

A simple FastAPI application that provides real-time monitoring data via WebSockets.
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import the monitoring module
from monitoring import setup_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitoring_frontend")

# Create FastAPI app
app = FastAPI(title="NextGen FinGPT Monitoring")

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="monitoring-frontend",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "frontend"}
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
        monitor.log_info("WebSocket client connected", 
                        connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
        monitor.log_info("WebSocket client disconnected", 
                        connections=len(self.active_connections))

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")

manager = ConnectionManager()

# System metrics collector
class MetricsCollector:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.task = None
    
    async def start(self):
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._collect_metrics())
        logger.info("Metrics collector started")
    
    async def stop(self):
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")
    
    async def _collect_metrics(self):
        while self.running:
            try:
                # Collect system metrics
                metrics = self._get_system_metrics()
                
                # Send metrics to all connected clients
                await manager.broadcast(json.dumps({
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": metrics
                }))
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                monitor.log_error(f"Error collecting metrics: {e}")
            
            await asyncio.sleep(self.interval)
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_total = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        return {
            "cpu": {
                "total_percent": cpu_total,
                "per_cpu_percent": cpu_percent
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
    
    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus metrics."""
        # CPU metrics
        monitor.set_gauge("cpu_utilization_percent", metrics["cpu"]["total_percent"], cpu="total")
        for i, percent in enumerate(metrics["cpu"]["per_cpu_percent"]):
            monitor.set_gauge("cpu_utilization_percent", percent, cpu=f"cpu{i}")
        
        # Memory metrics
        monitor.set_gauge("memory_usage_bytes", metrics["memory"]["total"], type="total")
        monitor.set_gauge("memory_usage_bytes", metrics["memory"]["used"], type="used")
        monitor.set_gauge("memory_usage_bytes", metrics["memory"]["free"], type="free")
        monitor.set_gauge("memory_usage_percent", metrics["memory"]["percent"], type="virtual")
        
        # Disk metrics
        monitor.set_gauge("disk_usage_bytes", metrics["disk"]["total"], device="root", mountpoint="/", type="total")
        monitor.set_gauge("disk_usage_bytes", metrics["disk"]["used"], device="root", mountpoint="/", type="used")
        monitor.set_gauge("disk_usage_bytes", metrics["disk"]["free"], device="root", mountpoint="/", type="free")
        monitor.set_gauge("disk_usage_percent", metrics["disk"]["percent"], device="root", mountpoint="/")
        
        # Network metrics
        monitor.set_gauge("network_io_bytes", metrics["network"]["bytes_sent"], interface="total", direction="sent")
        monitor.set_gauge("network_io_bytes", metrics["network"]["bytes_recv"], interface="total", direction="received")

# Create metrics collector
metrics_collector = MetricsCollector()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.on_event("startup")
async def startup_event():
    """Start the metrics collector when the application starts."""
    await metrics_collector.start()
    logger.info("Application started")
    monitor.log_info("Monitoring frontend started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the metrics collector when the application shuts down."""
    await metrics_collector.stop()
    logger.info("Application stopped")
    monitor.log_info("Monitoring frontend stopped")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the index.html page."""
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html")) as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from the client
            # This keeps the connection open
            data = await websocket.receive_text()
            
            # Process client messages if needed
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
