"""
Provides functions to generate synthetic data for end-to-end tests.
These functions are called by test modules when the 'synthetic' data source is selected.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_financial_market_data(num_records=100, tickers=['AAPL', 'MSFT', 'GOOGL']):
    """
    Generates synthetic financial market data (stock prices, volumes).
    """
    data = []
    start_date = datetime.now() - timedelta(days=num_records)
    for i in range(num_records):
        date = start_date + timedelta(days=i)
        for ticker in tickers:
            # Simulate price movement
            price = random.uniform(100.0, 1000.0)
            # Simulate volume
            volume = random.randint(100000, 5000000)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'open': price * random.uniform(0.98, 1.02),
                'high': price * random.uniform(0.99, 1.03),
                'low': price * random.uniform(0.97, 1.01),
                'close': price,
                'volume': volume
            })
    return pd.DataFrame(data)

def generate_sentiment_text_data(num_records=200, topics=['AAPL', 'MSFT', 'GOOGL', 'economy', 'technology']):
    """
    Generates synthetic text data for sentiment analysis (news headlines, social media posts).
    """
    headlines = [
        "{} stock {} after earnings report.",
        "Analyst upgrades {} to {} rating.",
        "New product launch from {}.",
        "Market reacts to {} news.",
        "{} announces partnership with another company."
    ]
    sentiments = ['positive', 'negative', 'neutral']
    data = []
    for i in range(num_records):
        topic = random.choice(topics)
        sentiment = random.choice(sentiments)
        headline_template = random.choice(headlines)
        # Simple fill-in-the-blanks for headlines
        if '{} stock' in headline_template:
             headline = headline_template.format(topic, sentiment)
        elif 'Analyst upgrades' in headline_template:
             rating = random.choice(['Buy', 'Hold', 'Sell'])
             headline = headline_template.format(topic, rating)
        elif 'New product launch' in headline_template:
             headline = headline_template.format(topic)
        elif 'Market reacts to' in headline_template:
             headline = headline_template.format(topic)
        elif 'announces partnership' in headline_template:
             headline = headline_template.format(topic)
        else:
             headline = headline_template.format(topic) # fallback

        data.append({
            'id': i + 1,
            'text': headline,
            'topic': topic,
            'sentiment': sentiment, # This is the 'ground truth' for testing
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat()
        })
    return pd.DataFrame(data)

def generate_time_series_data(num_points=365, series_name='synthetic_series'):
    """
    Generates synthetic time series data.
    """
    dates = [datetime.now() - timedelta(days=i) for i in range(num_points)]
    dates.reverse() # Make it chronological
    values = np.random.randn(num_points).cumsum() + 100 # Simple random walk
    data = {
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        series_name: values
    }
    return pd.DataFrame(data)

def generate_diverse_datasets():
    """
    Generates and returns a dictionary of diverse synthetic datasets.
    """
    datasets = {
        'financial_market_data': generate_financial_market_data(),
        'sentiment_text_data': generate_sentiment_text_data(),
        'time_series_data': generate_time_series_data()
        # Add more data generation functions here as needed for other MCP tools/models
    }
    return datasets

if __name__ == '__main__':
    # Example usage (optional, mainly for local testing of this script)
    # datasets = generate_diverse_datasets()
    # print("Generated Datasets:")
    # for name, df in datasets.items():
    #     print(f"\n--- {name} ---")
    #     print(df.head())
    #     print(f"Shape: {df.shape}")
    pass # This file should only contain generation logic, not execution for tests