"""
Stock Charts Generator for NextGen Platform

This module provides functionality to generate financial charts for stocks
and integrate them with the Netdata dashboard.

Features:
- Generate candlestick charts for stocks
- Display moving averages and technical indicators
- Create custom panels for the dashboard
- Real-time updates for active trading sessions

Dependencies:
- plotly (for chart generation)
- pandas (for data manipulation)
- requests (for API calls)
"""

import os
import time
import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monitoring.netdata_logger import NetdataLogger

class StockChartGenerator:
    """
    Generate financial charts for stocks and integrate with Netdata dashboard
    """

    def __init__(self, output_dir=None):
        """
        Initialize the chart generator

        Args:
            output_dir: Directory to save chart files (default: monitoring/dashboard/charts)
        """
        self.logger = NetdataLogger(component_name="stock-charts")

        # Set default output directory if not provided
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(current_dir, "dashboard", "charts")
        else:
            self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Stock chart generator initialized", output_dir=self.output_dir)

    def fetch_stock_data(self, symbol, timeframe="1d", limit=100, source="polygon"):
        """
        Fetch stock data from the specified source

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data (1m, 5m, 15m, 30m, 1h, 1d, 1w)
            limit: Number of data points to fetch
            source: Data source (polygon, yahoo, alpaca)

        Returns:
            Pandas DataFrame with OHLCV data
        """
        self.logger.info("Fetching stock data",
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        source=source)

        start_time = time.time()

        try:
            if source == "polygon":
                data = self._fetch_from_polygon(symbol, timeframe, limit)
            elif source == "yahoo":
                data = self._fetch_from_yahoo(symbol, timeframe, limit)
            elif source == "alpaca":
                data = self._fetch_from_alpaca(symbol, timeframe, limit)
            else:
                raise ValueError(f"Unsupported data source: {source}")

            duration = time.time() - start_time
            self.logger.timing("data_fetch_time_ms", duration * 1000)
            self.logger.info("Stock data fetched successfully",
                           symbol=symbol,
                           rows=len(data),
                           duration_ms=duration * 1000)

            return data

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Failed to fetch stock data",
                             symbol=symbol,
                             error=str(e),
                             duration_ms=duration * 1000)
            raise

    def _fetch_from_polygon(self, symbol, timeframe, limit):
        """Fetch data from Polygon.io"""
        # This would use the polygon_rest_mcp module in a real implementation
        # For now, we'll create some sample data
        dates = pd.date_range(end=datetime.datetime.now(), periods=limit)
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100 + i * 0.1 + (i % 10) for i in range(limit)],
            'high': [100 + i * 0.1 + 2 + (i % 10) for i in range(limit)],
            'low': [100 + i * 0.1 - 2 + (i % 10) for i in range(limit)],
            'close': [100 + i * 0.1 + 1 + (i % 10) for i in range(limit)],
            'volume': [1000000 + i * 1000 + (i % 100) * 10000 for i in range(limit)]
        })
        return data

    def _fetch_from_yahoo(self, symbol, timeframe, limit):
        """Fetch data from Yahoo Finance"""
        # This would use the yahoo_finance_mcp module in a real implementation
        return self._fetch_from_polygon(symbol, timeframe, limit)  # Use same sample data

    def _fetch_from_alpaca(self, symbol, timeframe, limit):
        """Fetch data from Alpaca"""
        # This would use the alpaca_mcp module in a real implementation
        return self._fetch_from_polygon(symbol, timeframe, limit)  # Use same sample data

    def calculate_indicators(self, data):
        """
        Calculate technical indicators for the stock data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()

        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Calculate MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def create_candlestick_chart(self, data, symbol, timeframe, include_volume=True,
                               include_indicators=True, output_file=None):
        """
        Create a candlestick chart for the stock data

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            timeframe: Timeframe of the data
            include_volume: Whether to include volume subplot
            include_indicators: Whether to include technical indicators
            output_file: File to save the chart to (default: auto-generated)

        Returns:
            Path to the saved chart file
        """
        self.logger.info("Creating candlestick chart",
                        symbol=symbol,
                        timeframe=timeframe,
                        include_volume=include_volume,
                        include_indicators=include_indicators)

        start_time = time.time()

        try:
            # Calculate indicators if requested
            if include_indicators:
                df = self.calculate_indicators(data)
            else:
                df = data.copy()

            # Determine number of rows for subplots
            rows = 1
            if include_volume:
                rows += 1
            if include_indicators:
                rows += 1  # For MACD

            # Create subplot grid
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=(f"{symbol} {timeframe}", "Volume", "MACD")[:rows]
            )

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="OHLC"
                ),
                row=1, col=1
            )

            # Add moving averages if indicators are included
            if include_indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['sma_20'],
                        line=dict(color='blue', width=1),
                        name="SMA 20"
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['sma_50'],
                        line=dict(color='orange', width=1),
                        name="SMA 50"
                    ),
                    row=1, col=1
                )

            # Add volume subplot if requested
            current_row = 2
            if include_volume:
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name="Volume",
                        marker=dict(color='rgba(0, 0, 255, 0.5)')
                    ),
                    row=current_row, col=1
                )
                current_row += 1

            # Add MACD subplot if indicators are included
            if include_indicators and current_row <= rows:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['macd'],
                        line=dict(color='blue', width=1),
                        name="MACD"
                    ),
                    row=current_row, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['signal'],
                        line=dict(color='red', width=1),
                        name="Signal"
                    ),
                    row=current_row, col=1
                )

            # Update layout
            fig.update_layout(
                title=f"{symbol} Stock Chart ({timeframe})",
                xaxis_title="Date",
                yaxis_title="Price",
                height=800,
                width=1000,
                showlegend=True
            )

            # Remove rangeslider
            fig.update_layout(xaxis_rangeslider_visible=False)

            # Generate output file name if not provided
            if output_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.output_dir, f"{symbol}_{timeframe}_{timestamp}.html")

            # Save the chart
            fig.write_html(output_file)

            duration = time.time() - start_time
            self.logger.timing("chart_creation_time_ms", duration * 1000)
            self.logger.info("Chart created successfully",
                           symbol=symbol,
                           output_file=output_file,
                           duration_ms=duration * 1000)

            return output_file

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Failed to create chart",
                             symbol=symbol,
                             error=str(e),
                             duration_ms=duration * 1000)
            raise

    def create_multi_stock_chart(self, symbols, timeframe="1d", limit=100,
                               normalize=True, output_file=None):
        """
        Create a chart comparing multiple stocks

        Args:
            symbols: List of stock symbols
            timeframe: Timeframe for data
            limit: Number of data points
            normalize: Whether to normalize prices to percentage change
            output_file: File to save the chart to

        Returns:
            Path to the saved chart file
        """
        self.logger.info("Creating multi-stock chart",
                        symbols=symbols,
                        timeframe=timeframe,
                        normalize=normalize)

        start_time = time.time()

        try:
            fig = go.Figure()

            # Fetch data for each symbol
            for symbol in symbols:
                data = self.fetch_stock_data(symbol, timeframe, limit)

                if normalize:
                    # Calculate percentage change from first data point
                    first_close = data['close'].iloc[0]
                    data['normalized'] = (data['close'] / first_close - 1) * 100
                    y_values = data['normalized']
                    y_axis_title = "Percentage Change (%)"
                else:
                    y_values = data['close']
                    y_axis_title = "Price"

                # Add line for this symbol
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=y_values,
                        mode='lines',
                        name=symbol
                    )
                )

            # Update layout
            fig.update_layout(
                title=f"Stock Comparison ({timeframe})",
                xaxis_title="Date",
                yaxis_title=y_axis_title,
                height=600,
                width=1000,
                showlegend=True
            )

            # Generate output file name if not provided
            if output_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                symbols_str = "_".join(symbols)
                if len(symbols_str) > 30:
                    symbols_str = symbols_str[:30] + "..."
                output_file = os.path.join(self.output_dir, f"comparison_{symbols_str}_{timestamp}.html")

            # Save the chart
            fig.write_html(output_file)

            duration = time.time() - start_time
            self.logger.timing("multi_chart_creation_time_ms", duration * 1000)
            self.logger.info("Multi-stock chart created successfully",
                           symbols=symbols,
                           output_file=output_file,
                           duration_ms=duration * 1000)

            return output_file

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Failed to create multi-stock chart",
                             symbols=symbols,
                             error=str(e),
                             duration_ms=duration * 1000)
            raise

# Example usage
if __name__ == "__main__":
    # Create chart generator
    generator = StockChartGenerator()

    # Generate a single stock chart
    data = generator.fetch_stock_data("AAPL", timeframe="1d", limit=100)
    chart_file = generator.create_candlestick_chart(data, "AAPL", "1d")
    print(f"Chart saved to: {chart_file}")

    # Generate a multi-stock comparison chart
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    comparison_file = generator.create_multi_stock_chart(symbols, normalize=True)
    print(f"Comparison chart saved to: {comparison_file}")