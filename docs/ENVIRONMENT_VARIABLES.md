# Environment Variables in NextGen

This document details the environment variables required for the NextGen trading platform.

## Configuration Setup

The NextGen platform uses environment variables for configuration, including API keys, connection details, and behavioral settings. We use `python-dotenv` to load these variables from a `.env` file.

### Setting Up Your Environment

1. Create a `.env` file in the project root directory
2. Copy the template from `.env.example` 
3. Fill in your actual values
4. Never commit your actual `.env` file to version control

## Critical API Keys

The following API keys are critical for the platform's core functionality:

| Environment Variable | Service | Required For | Format Example |
|---------------------|---------|--------------|----------------|
| `OPENROUTER_API_KEY` | OpenRouter | LLM functionality for AutoGen orchestrator | `sk-or-v1-...` |
| `POLYGON_API_KEY` | Polygon.io | Market data retrieval | `PK_...` |
| `ALPACA_API_KEY` | Alpaca | Trading operations | `AK...` |
| `ALPACA_SECRET_KEY` | Alpaca | Trading operations | - |

## Market Data Services

| Environment Variable | Service | Description |
|---------------------|---------|-------------|
| `UNUSUAL_WHALES_API_KEY` | Unusual Whales | Options flow and unusual activity data |
| `YAHOO_FINANCE_API_KEY` | Yahoo Finance | General market data (fallback) |

## Social & News Data 

| Environment Variable | Service | Description |
|---------------------|---------|-------------|
| `REDDIT_CLIENT_ID` | Reddit API | Social sentiment analysis |
| `REDDIT_CLIENT_SECRET` | Reddit API | Social sentiment analysis |
| `REDDIT_USERNAME` | Reddit API | Social sentiment analysis |
| `REDDIT_PASSWORD` | Reddit API | Social sentiment analysis |
| `REDDIT_USER_AGENT` | Reddit API | Identity for Reddit API requests |

## Infrastructure

| Environment Variable | Service | Description |
|---------------------|---------|-------------|
| `REDIS_HOST` | Redis | Host for Redis server |
| `REDIS_PORT` | Redis | Port for Redis server |
| `REDIS_DB` | Redis | Database number |
| `REDIS_PASSWORD` | Redis | Password for Redis server |
| `INFLUXDB_URL` | InfluxDB | URL for InfluxDB |
| `INFLUXDB_TOKEN` | InfluxDB | Authentication token |
| `INFLUXDB_ORG` | InfluxDB | Organization name |
| `INFLUXDB_BUCKET` | InfluxDB | Bucket name for data |

## Monitoring

| Environment Variable | Service | Description |
|---------------------|---------|-------------|
| `GRAFANA_URL` | Grafana | URL for Grafana dashboard |
| `GRAFANA_USER` | Grafana | Grafana username |
| `GRAFANA_PASSWORD` | Grafana | Grafana password |
| `LOKI_URL` | Loki | URL for Loki log aggregation |
| `PROMETHEUS_METRICS_PORT` | Prometheus | Port to expose metrics |

## Others

| Environment Variable | Service | Description |
|---------------------|---------|-------------|
| `NGROK_AUTH_TOKEN` | Ngrok | Authentication token for tunneling |
| `NGROK_API_KEY` | Ngrok | API key for Ngrok |
| `GITHUB_USERNAME` | GitHub | Username for GitHub operations |
| `GITHUB_EMAIL` | GitHub | Email for GitHub operations |
| `GITHUB_TOKEN` | GitHub | Personal access token |

## FinGPT Configuration

These settings control the behavior of the FinGPT orchestrator:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `FINGPT_MODEL_PATH` | Model path/name to use | `gpt-4.1` |
| `FINGPT_MAX_TOKENS` | Maximum tokens for completion | `1024` |
| `FINGPT_TEMPERATURE` | Randomness/creativity setting | `0.1` |
| `FINGPT_CONTEXT_WINDOW_SIZE` | Context window size | `4096` |
| `FINGPT_CACHE_TTL` | Cache time-to-live in seconds | `300` |
| `FINGPT_CONFIDENCE_THRESHOLD` | Confidence threshold for actions | `0.7` |

## Risk Parameters

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `FINGPT_MAX_POSITION_SIZE` | Maximum position size as fraction | `0.05` |
| `FINGPT_MAX_POSITIONS` | Maximum concurrent positions | `10` |
| `FINGPT_MAX_TRADES_PER_DAY` | Maximum trades per day | `20` |
| `FINGPT_STOP_LOSS_PCT` | Default stop loss percentage | `0.02` |
| `FINGPT_TAKE_PROFIT_PCT` | Default take profit percentage | `0.05` |

## Validation and Error Handling

The system validates critical API keys on startup and logs appropriate warnings if they are missing or incorrectly formatted. Components that absolutely require certain API keys will throw explicit errors when those keys are missing, preventing unexpected behavior or silent failures.

## Troubleshooting Missing Keys

If you see error messages about missing API keys:

1. Check that your `.env` file exists in the project root
2. Verify the correct variable names are being used
3. Ensure no typos or extra whitespace in your keys
4. Validate that your API keys are still active with the service provider

## Environment Management Functions

The `utils.env_loader` module provides these functions for working with environment variables:

- `load_environment()`: Load variables from .env file
- `get_api_key(service_name)`: Get API key for a specific service
- `get_env(key, default)`: Get any environment variable 
- `validate_api_keys()`: Validate all critical API keys