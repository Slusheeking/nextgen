# FinGPT AI Day Trading System

A comprehensive AI-powered trading system orchestrated by language models, built on the FinGPT framework for financial market analysis and trading execution.

## System Architecture

The FinGPT AI Day Trading System is organized as a modular, event-driven architecture with several key components:

1. **Data Ingestion Layer**: Collects market data from multiple sources
2. **Stock Selection Layer**: Identifies trading candidates through multi-tier filtering
3. **Analysis Layer**: Processes data through specialized components (NLP, forecasting, RAG)
4. **Decision Layer**: Orchestrates analysis results into actionable trading decisions
5. **Execution Layer**: Implements trades with dynamic position sizing
6. **Monitoring Layer**: Tracks performance and system health

## Model Sources and Data Providers

### FinGPT Models
- **FinGPT Base Models**: https://github.com/AI4Finance-Foundation/FinGPT
  - FinGPT-v3-llama2-7b: https://huggingface.co/financial-llm/finllama-7b
  - FinGPT-RAG: https://huggingface.co/FinGPT/fingpt-rag_llama2-7b 
  - FinGPT-Forecaster: https://huggingface.co/FinGPT/fingpt-forecaster_llama2-13b

### External LLM Providers
- **OpenAI API**: https://platform.openai.com/
  - GPT-4 and GPT-3.5 Turbo for high-complexity financial analysis
- **Anthropic Claude API**: https://www.anthropic.com/claude
  - Claude 3 Opus for in-depth financial document analysis
- **Mistral AI**: https://mistral.ai/
  - Mistral Large for additional financial analysis capabilities

### Market Data Sources
- **Polygon.io**: https://polygon.io/
  - Historical and real-time market data via REST and WebSocket APIs
  - Options data and market indicators
- **Alpha Vantage**: https://www.alphavantage.co/
  - Fundamental data and alternative datasets
- **Yahoo Finance**: https://finance.yahoo.com/
  - Earnings reports, analyst recommendations, news
- **Unusual Whales**: https://unusualwhales.com/
  - Options flow and unusual activity detection
- **Reddit API**: https://www.reddit.com/dev/api/
  - Social sentiment and retail investor activity

### Financial Knowledge Sources
- **SEC EDGAR**: https://www.sec.gov/edgar
  - Regulatory filings and corporate disclosures
- **Federal Reserve Economic Data (FRED)**: https://fred.stlouisfed.org/
  - Macroeconomic indicators and financial data

## Core Components

### 1. fingpt/fingpt_orchestrator/orchestrator_model.py
- **Purpose**: Central brain of the entire system
- **Functions**:
  - Coordinate all other components
  - Manage the overall trading workflow
  - Handle event-driven communication
  - Maintain global state awareness
  - Adapt to changing market conditions
  - Make high-level strategic decisions

### 2. fingpt/fingpt_selection/selection_model.py
- **Purpose**: Identify potential trading candidates
- **Functions**:
  - Initial universe generation
  - Multi-tier stock screening
  - Apply technical and fundamental filters
  - Integrate with backtrader for screening
  - Produce final list of stocks for analysis

### 3. fingpt/fingpt_forcaster/forcaster_model.py
- **Purpose**: Time series forecasting and price prediction
- **Functions**:
  - Analyze price trends and patterns
  - Generate short-term price forecasts
  - Identify support/resistance levels
  - Detect trend strength and direction
  - Predict potential breakouts/breakdowns

### 4. fingpt/fingpt_finnlp/finnlp_model.py
- **Purpose**: Comprehensive financial text processing and analysis
- **Functions**:
  - Process news articles and social media
  - Score sentiment on financial texts
  - Detect sentiment shifts and trends
  - Identify key sentiment drivers
  - Correlate sentiment with price action
  - Extract structured data from financial documents
  - Process earnings reports and SEC filings
  - Identify key financial metrics from text
  - Extract entity relationships
  - Process financial news and research

### 5. fingpt/fingpt_rag/rag_model.py
- **Purpose**: Retrieval-augmented generation for financial context
- **Functions**:
  - Access knowledge bases for context
  - Incorporate historical patterns and precedents
  - Provide factual grounding for decisions
  - Integrate domain knowledge into analysis
  - Connect current conditions with historical context

### 6. fingpt/fingpt_execution/execution_model.py
- **Purpose**: Intelligent trade execution
- **Functions**:
  - Determine optimal execution strategy
  - Implement dynamic position sizing
  - Manage execution timing
  - Minimize market impact
  - Connect with Alpaca for execution
  - Monitor execution quality

### 7. fingpt/fingpt_order/order_managment_model.py
- **Purpose**: Manage active orders and positions
- **Functions**:
  - Track open positions and orders
  - Implement stop-loss and take-profit logic
  - Monitor position performance
  - Handle order lifecycle management
  - Calculate portfolio metrics
  - Implement risk limits and checks

### 8. fingpt/fingpt_bench/bench_model.py
- **Purpose**: Benchmark and evaluate system performance
- **Functions**:
  - Track performance metrics
  - Evaluate strategy effectiveness
  - Compare against baselines
  - Generate performance reports
  - Identify improvement opportunities
  - Validate model predictions

### 9. fingpt/fingpt_lora/lora_model.py
- **Purpose**: Low-rank adaptation for LLMs
- **Functions**:
  - Fine-tune base LLMs for financial tasks
  - Adapt models with minimal compute resources
  - Maintain specialized financial adaptations
  - Enable efficient model updates
  - Store and manage LoRA weights

## Data Sources and Processing

### 10. data/polygon_rest.py
- **Purpose**: Interface with Polygon.io REST API
- **Functions**:
  - Fetch historical market data
  - Retrieve reference data (splits, dividends)
  - Get fundamental company data
  - Access market indices and indicators
  - Handle API rate limiting and pagination

### 11. data/polygon_ws.py
- **Purpose**: Interface with Polygon.io WebSocket API
- **Functions**:
  - Stream real-time market data
  - Process live trades and quotes
  - Maintain persistent connections
  - Handle reconnection logic
  - Buffer and process streaming data

### 12. data/yahoo_finance.py
- **Purpose**: Interface with Yahoo Finance
- **Functions**:
  - Access fundamental company data
  - Retrieve historical price data
  - Get options chains information
  - Access financial statements
  - Fetch analyst recommendations

### 13. data/reddit.py
- **Purpose**: Interface with Reddit for sentiment data
- **Functions**:
  - Monitor financial subreddits
  - Track trending tickers
  - Gather sentiment from user posts
  - Identify emerging narratives
  - Detect unusual activity spikes

### 14. data/unusual_whales.py
- **Purpose**: Interface with Unusual Whales API
- **Functions**:
  - Track unusual options activity
  - Monitor large block trades
  - Detect potential informed trading
  - Identify unusual volume patterns
  - Track institutional money flow

### 15. data/data_preprocessor.py
- **Purpose**: Process and prepare data for analysis
- **Functions**:
  - Clean and normalize market data
  - Calculate technical indicators
  - Standardize data formats
  - Handle missing data
  - Prepare time series for models
  - Transform data for model inputs

## Trading Integration

### 16. alpaca/alpaca_api.py
- **Purpose**: Interface with Alpaca trading platform
- **Functions**:
  - Execute buy/sell orders
  - Manage positions and orders
  - Stream market data
  - Access account information
  - Handle authentication and API limits
  - Process execution reports

## Infrastructure Components

### 17. redis/redis_manager.py
- **Purpose**: Manage Redis for caching and messaging
- **Functions**:
  - Cache frequently used data
  - Implement pub/sub for event distribution
  - Store temporary state information
  - Maintain distributed state
  - Implement distributed locking
  - Enable fast data access

### 18. influxdb/influxdb_manager.py
- **Purpose**: Manage time series data storage
- **Functions**:
  - Store historical market data
  - Track system performance metrics
  - Enable time series analytics
  - Support visualization dashboards
  - Implement data retention policies
  - Handle time-based aggregations

### 19. loki/loki_manager.py
- **Purpose**: Centralized logging
- **Functions**:
  - Collect logs from all components
  - Enable structured logging
  - Support log querying and analysis
  - Implement log retention policies
  - Alert on error patterns
  - Track system health

### 20. prometheus/prometheus_manager.py
- **Purpose**: Metrics collection and monitoring
- **Functions**:
  - Track system performance metrics
  - Monitor resource utilization
  - Implement alerting based on thresholds
  - Support metrics visualization
  - Enable system health monitoring
  - Provide performance insights

## Configuration and Environment

### 21. .env and .env.example
- **Purpose**: Store environment configuration
- **Functions**:
  - Manage API keys and credentials
  - Configure service endpoints
  - Set operational parameters
  - Specify environment-specific settings
  - Control debugging options

### 22. systemd/requirements.txt
- **Purpose**: Define systemd service dependencies
- **Functions**:
  - Support system service management
  - Enable auto-restart functionality
  - Provide system-level logging
  - Support startup sequencing
  - Handle dependency management

## System Workflow

The system operates through an event-driven architecture with the following key workflow:

1. **Stock Discovery**: The stock selection component identifies potential trading candidates through multi-tier filtering
2. **Data Gathering**: Data connectors collect relevant information for the selected stocks
3. **Analysis**: Multiple analysis components process the gathered data:
   - FinNLP analyzes news, social media, and financial documents
   - Forecaster generates price predictions and identifies patterns
   - RAG provides historical context and precedents
4. **Decision Making**: The orchestrator integrates all analysis and makes trading decisions
5. **Execution**: The execution component implements trades with dynamic sizing
6. **Position Management**: The order management component tracks and manages active positions
7. **Performance Analysis**: The benchmarking component measures strategy effectiveness and provides feedback

## Integration Points

- **Redis** serves as the primary message bus and caching layer
- **InfluxDB** provides time-series storage for market data and performance metrics
- **Prometheus** and **Loki** handle monitoring and logging
- **Alpaca** serves as the execution endpoint for trades

## Dependencies

- **FinGPT Framework**: AI4Finance Foundation's financial LLM framework
- **LLM Models**: Large language models for financial text analysis and decision making
- **Market Data Providers**: Polygon.io, Yahoo Finance, Reddit, Unusual Whales
- **Execution Brokers**: Alpaca
- **Infrastructure**: Redis, InfluxDB, Loki, Prometheus

## Development and Deployment

The system is designed to run as a collection of services, each handling a specific part of the trading workflow. Components communicate through a well-defined event system, allowing for distributed deployment and scaling of individual components as needed.
