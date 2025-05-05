"""
Example usage of NextGen Monitoring System

This file demonstrates how to use the various components of the NextGen monitoring system:
1. NetdataLogger for logging and basic metrics
2. SystemMetricsCollector for enhanced system and hardware metrics
3. StockChartGenerator for financial charts
4. 24/7 monitoring service setup and management
"""

import time
import random
import os
import subprocess
import threading
from dotenv import load_dotenv
load_dotenv()

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector
from monitoring.stock_charts import StockChartGenerator

# Example 1: Basic logging and metrics with NetdataLogger
def example_basic_logging():
    print("\n--- Example 1: Basic Logging and Metrics ---")
    
    # Create a logger for this component
    logger = NetdataLogger(component_name="example-basic")
    
    # Log an info message
    logger.info("Basic example started")
    
    # Send a gauge metric
    logger.gauge("example_value", 42.5)
    
    # Log with additional structured data
    logger.info("Processing data", items=100, source="database")
    
    # Log an error with exception information
    try:
        result = 1 / 0
    except Exception as e:
        logger.error("Calculation failed", error=str(e), operation="division")
    
    # Log different levels
    logger.debug("This is a debug message", detail="extra info")
    logger.warning("This is a warning", reason="example")
    logger.critical("This is a critical message", impact="none, just an example")
    
    logger.info("Basic example completed")
    
    print("Basic logging example completed. Check logs directory for log files.")
    print("Metrics have been sent to Netdata and can be viewed at http://localhost:19999")


# Example 2: Enhanced System Metrics Collection
def example_system_metrics():
    print("\n--- Example 2: Enhanced System Metrics Collection ---")
    
    # Create a logger
    logger = NetdataLogger(component_name="system-metrics-example")
    
    # Create the system metrics collector
    collector = SystemMetricsCollector(logger)
    
    # Start collecting metrics
    collector.start()
    
    print("System metrics collector started")
    print("Collecting CPU, memory, disk, network, and GPU metrics (if available)")
    print("Metrics will be sent to Netdata every 5 seconds")
    
    # Run for 30 seconds
    for i in range(6):
        print(f"Collecting metrics... ({i+1}/6)")
        
        # Simulate some CPU and memory load
        if i == 2:
            print("Simulating CPU load...")
            cpu_load_thread = threading.Thread(target=simulate_cpu_load, args=(5,))
            cpu_load_thread.start()
        
        if i == 4:
            print("Simulating memory load...")
            memory_load_thread = threading.Thread(target=simulate_memory_load)
            memory_load_thread.start()
        
        time.sleep(5)
    
    # Stop the collector
    collector.stop()
    
    print("System metrics collection stopped")
    print("Check Netdata dashboard at http://localhost:19999 to view the collected metrics")


# Example 3: Stock Chart Generation
def example_stock_charts():
    print("\n--- Example 3: Stock Chart Generation ---")
    
    # Create a chart generator
    generator = StockChartGenerator()
    
    # Example 1: Generate a single stock chart
    print("Generating a candlestick chart for AAPL...")
    data = generator.fetch_stock_data("AAPL", timeframe="1d", limit=100)
    chart_file = generator.create_candlestick_chart(data, "AAPL", "1d")
    print(f"Chart saved to: {chart_file}")
    
    # Example 2: Generate a multi-stock comparison chart
    print("Generating a comparison chart for AAPL, MSFT, GOOGL, and AMZN...")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    comparison_file = generator.create_multi_stock_chart(symbols, normalize=True)
    print(f"Comparison chart saved to: {comparison_file}")
    
    print("Stock charts generated successfully")
    print("View the charts by opening the HTML files in a web browser")
    print("Or access them through the dashboard at http://localhost:8080")


# Example 4: 24/7 Monitoring Service Management
def example_service_management():
    print("\n--- Example 4: 24/7 Monitoring Service Management ---")
    
    # Check if running as root (required for systemctl commands)
    if os.geteuid() != 0:
        print("This example requires root privileges to manage system services")
        print("Run this script with sudo to see the service management example")
        return
    
    # Check service status
    print("Checking monitoring service status...")
    try:
        status = subprocess.run(
            ["systemctl", "status", "nextgen-monitoring.service"],
            capture_output=True,
            text=True
        )
        print(status.stdout)
    except Exception as e:
        print(f"Error checking service status: {e}")
    
    # Demonstrate how to restart the service
    print("Demonstrating service restart...")
    try:
        restart = subprocess.run(
            ["systemctl", "restart", "nextgen-monitoring.service"],
            capture_output=True,
            text=True
        )
        print("Service restarted successfully")
    except Exception as e:
        print(f"Error restarting service: {e}")
    
    # Show how to view logs
    print("Showing recent service logs...")
    try:
        logs = subprocess.run(
            ["journalctl", "-u", "nextgen-monitoring.service", "-n", "10"],
            capture_output=True,
            text=True
        )
        print(logs.stdout)
    except Exception as e:
        print(f"Error viewing logs: {e}")
    
    print("Service management example completed")
    print("In a real environment, you would use these commands to manage the 24/7 monitoring service")


# Example 5: Integration with a Trading Model
class MonitoredTradingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.logger = NetdataLogger(component_name=f"trading-model-{model_name}")
        
        # Start system metrics collection
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.metrics_collector.start()
        
        # Initialize chart generator
        self.chart_generator = StockChartGenerator()
        
        self.logger.info("Trading model initialized", model_name=model_name)
    
    def analyze_stock(self, symbol, timeframe="1d"):
        self.logger.info("Analyzing stock", symbol=symbol, timeframe=timeframe)
        
        start_time = time.time()
        
        # Fetch stock data
        try:
            data = self.chart_generator.fetch_stock_data(symbol, timeframe=timeframe)
            self.logger.info("Stock data fetched", symbol=symbol, data_points=len(data))
            
            # Generate chart
            chart_file = self.chart_generator.create_candlestick_chart(
                data, symbol, timeframe, include_volume=True, include_indicators=True
            )
            self.logger.info("Chart generated", symbol=symbol, chart_file=chart_file)
            
            # Simulate analysis
            time.sleep(random.uniform(0.5, 1.5))
            
            # Generate random analysis results
            sentiment = random.choice(["bullish", "bearish", "neutral"])
            confidence = random.uniform(0.6, 0.95)
            target_price = data['close'].iloc[-1] * (1 + random.uniform(-0.1, 0.2))
            
            duration = time.time() - start_time
            
            # Record metrics
            self.logger.timing("analysis_time_ms", duration * 1000)
            self.logger.gauge(f"sentiment_{sentiment}", 1)
            self.logger.gauge("analysis_confidence", confidence)
            
            self.logger.info("Analysis completed", 
                           symbol=symbol,
                           sentiment=sentiment,
                           confidence=confidence,
                           target_price=target_price,
                           duration_ms=duration * 1000)
            
            return {
                "symbol": symbol,
                "sentiment": sentiment,
                "confidence": confidence,
                "target_price": target_price,
                "chart_file": chart_file
            }
            
        except Exception as e:
            self.logger.error("Analysis failed", symbol=symbol, error=str(e))
            raise
    
    def shutdown(self):
        self.logger.info("Trading model shutting down")
        self.metrics_collector.stop()


# Helper functions for system load simulation
def simulate_cpu_load(duration=5):
    """Simulate CPU load by performing calculations"""
    end_time = time.time() + duration
    while time.time() < end_time:
        # Perform CPU-intensive calculations
        [i**2 for i in range(10000)]
        [float(i)/3.14159 for i in range(10000)]

def simulate_memory_load():
    """Simulate memory load by allocating a large list"""
    # Allocate ~100MB of memory
    large_list = [i for i in range(10000000)]
    time.sleep(2)
    # Free the memory
    del large_list


# Run the examples if this file is executed directly
if __name__ == "__main__":
    print("Running NextGen Monitoring System Examples...")
    
    # Example 1: Basic logging and metrics
    example_basic_logging()
    
    # Example 2: Enhanced system metrics
    example_system_metrics()
    
    # Example 3: Stock charts
    example_stock_charts()
    
    # Example 4: Service management (requires root)
    if os.geteuid() == 0:
        example_service_management()
    else:
        print("\n--- Example 4: Service Management (Skipped) ---")
        print("This example requires root privileges. Run with sudo to see it.")
    
    # Example 5: Integrated trading model
    print("\n--- Example 5: Integrated Trading Model ---")
    model = MonitoredTradingModel("example-model")
    
    try:
        # Analyze a few stocks
        stocks = ["AAPL", "MSFT", "GOOGL"]
        for symbol in stocks:
            result = model.analyze_stock(symbol)
            print(f"Analysis for {symbol}: {result['sentiment']} (confidence: {result['confidence']:.2f})")
            print(f"Target price: ${result['target_price']:.2f}")
            print(f"Chart saved to: {result['chart_file']}")
    finally:
        # Ensure we shut down properly
        model.shutdown()
    
    print("\nAll examples completed.")
    print("You can access the monitoring dashboard at http://localhost:8080")
    print("Netdata metrics are available at http://localhost:19999")
    print("Log files are stored in the logs directory")
