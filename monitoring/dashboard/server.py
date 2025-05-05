#!/usr/bin/env python3
"""
Simple HTTP server for NextGen monitoring dashboard, including stock chart API
"""
import http.server
import socketserver
import os
import sys
import json
from urllib.parse import urlparse, parse_qs

# Add project root to path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import the stock chart generator
from monitoring.stock_charts import StockChartGenerator

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(DIRECTORY, "charts")

# Ensure charts directory exists
os.makedirs(CHARTS_DIR, exist_ok=True)

# Initialize chart generator
chart_generator = StockChartGenerator(output_dir=CHARTS_DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize with the dashboard directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        """Handle GET requests, including API calls"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        if path == '/api/stock_chart':
            self.handle_stock_chart_api(query_params)
        elif path == '/api/multi_stock_chart':
            self.handle_multi_stock_chart_api(query_params)
        elif path.startswith('/charts/'):
            # Serve chart files from the charts subdirectory
            chart_file = os.path.basename(path)
            chart_path = os.path.join(CHARTS_DIR, chart_file)
            try:
                if os.path.exists(chart_path) and os.path.isfile(chart_path):
                    with open(chart_path, 'rb') as f:
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(f.read())
                else:
                    self.send_error(404, f"Chart file not found: {chart_file}")
            except Exception as e:
                self.send_error(500, f"Error serving chart file: {str(e)}")
        else:
            # Handle regular file requests
            super().do_GET()

    def handle_stock_chart_api(self, query_params):
        """Handle API requests for stock charts"""
        try:
            # Get parameters
            symbol = query_params.get('symbol', ['AAPL'])[0]
            timeframe = query_params.get('timeframe', ['1d'])[0]
            include_volume = query_params.get('volume', ['true'])[0].lower() == 'true'
            include_indicators = query_params.get('indicators', ['true'])[0].lower() == 'true'
            
            # Generate a unique filename for this chart
            timestamp = self.get_timestamp()
            output_file = os.path.join(CHARTS_DIR, f"{symbol}_{timeframe}_{timestamp}.html")
            
            # Fetch data and create chart
            data = chart_generator.fetch_stock_data(symbol, timeframe)
            chart_file = chart_generator.create_candlestick_chart(
                data, symbol, timeframe, 
                include_volume=include_volume,
                include_indicators=include_indicators,
                output_file=output_file
            )
            
            # Get the relative URL for the chart
            chart_url = f"/charts/{os.path.basename(chart_file)}"
            
            # Create iframe HTML to embed the chart
            iframe_html = f'<iframe src="{chart_url}" width="100%" height="600px" frameborder="0"></iframe>'
            
            # Return success response
            self.send_json_response({
                'success': True,
                'chart_url': chart_url,
                'html': iframe_html
            })
            
        except Exception as e:
            # Return error response
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, status_code=500)

    def handle_multi_stock_chart_api(self, query_params):
        """Handle API requests for multi-stock comparison charts"""
        try:
            # Get parameters
            symbols_str = query_params.get('symbols', ['AAPL,MSFT,GOOGL'])[0]
            symbols = [s.strip() for s in symbols_str.split(',')]
            timeframe = query_params.get('timeframe', ['1d'])[0]
            normalize = query_params.get('normalize', ['true'])[0].lower() == 'true'
            
            # Generate a unique filename for this chart
            timestamp = self.get_timestamp()
            symbols_short = "_".join(symbols[:3])
            if len(symbols) > 3:
                symbols_short += "_etc"
            output_file = os.path.join(CHARTS_DIR, f"comparison_{symbols_short}_{timestamp}.html")
            
            # Create chart
            chart_file = chart_generator.create_multi_stock_chart(
                symbols, timeframe, normalize=normalize, output_file=output_file
            )
            
            # Get the relative URL for the chart
            chart_url = f"/charts/{os.path.basename(chart_file)}"
            
            # Create iframe HTML to embed the chart
            iframe_html = f'<iframe src="{chart_url}" width="100%" height="600px" frameborder="0"></iframe>'
            
            # Return success response
            self.send_json_response({
                'success': True,
                'chart_url': chart_url,
                'html': iframe_html
            })
            
        except Exception as e:
            # Return error response
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, status_code=500)

    def send_json_response(self, data, status_code=200):
        """Send a JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_timestamp(self):
        """Get a timestamp string for filenames"""
        import datetime
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving dashboard at http://localhost:{PORT}")
            print(f"Stock charts will be saved to {CHARTS_DIR}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()