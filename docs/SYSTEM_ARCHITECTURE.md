# NextGen AI Trading System

A comprehensive AI-powered trading system orchestrated by language models, built on a modular architecture using MCP (Model Context Protocol) for efficient tool integration and AutoGen for agent orchestration.

## System Architecture

The NextGen AI Trading System is organized as a modular, event-driven architecture with several key components:

1. **Data Ingestion Layer**: Collects market data from multiple sources through MCP data tools
2. **Stock Selection Layer**: Identifies trading candidates through multi-tier filtering
3. **Analysis Layer**: Processes data through specialized components and MCP analysis tools
4. **Decision Layer**: Orchestrates analysis results into actionable trading decisions
5. **Execution Layer**: Implements trades with dynamic position sizing via Alpaca MCP
6. **Monitoring Layer**: Tracks performance and system health

## Model Sources and Data Providers

### LLM Providers
- **OpenRouter API**: https://openrouter.ai/
  - Anthropic Claude 3 Opus for in-depth financial document analysis
  - Meta Llama 3 70B for complex financial reasoning
  - Google Gemini Pro for additional financial analysis capabilities

### Market Data Sources
- **Polygon.io**: https://polygon.io/
  - Historical and real-time market data via REST and WebSocket APIs
  - Options data and market indicators via polygon_rest_mcp and polygon_ws_mcp
- **Yahoo Finance**: https://finance.yahoo.com/
  - Earnings reports, analyst recommendations, news via yahoo_finance_mcp
- **Unusual Whales**: https://unusualwhales.com/
  - Options flow and unusual activity detection via unusual_whales_mcp
- **Reddit**: https://www.reddit.com/
  - Social sentiment and retail investor activity via reddit_mcp

### Trading Integration
- **Alpaca Markets**: https://alpaca.markets/
  - Order execution and position management via alpaca_mcp

### Data Storage
- **Redis**: In-memory data store for real-time data and coordination
  - Integrated via redis_mcp for caching and messaging
- **Vector Database**: Local vectordb for embeddings storage
  - Enables efficient similarity search for financial data

## Core Components

### 1. nextgen_models/autogen_orchestrator/autogen_model.py
- **Purpose**: Central brain of the entire system using Microsoft's AutoGen framework
- **Functions**:
  - Coordinate all other components
  - Manage the overall trading workflow
  - Handle event-driven communication
  - Maintain global state awareness
  - Adapt to changing market conditions
  - Make high-level strategic decisions
  - Register specialized agents for different tasks
  - Proxy MCP tool access to agents

### 2. nextgen_models/nextgen_select/select_model.py
- **Purpose**: Identify potential trading candidates
- **Functions**:
  - Initial universe generation
  - Multi-tier stock screening
  - Apply technical and fundamental filters
  - Integrate with technical indicators via MCP
  - Produce final list of stocks for analysis

### 3. nextgen_models/nextgen_market_analysis/market_analysis_model.py
- **Purpose**: Time series forecasting and price prediction
- **Functions**:
  - Analyze price trends and patterns
  - Generate short-term price forecasts
  - Identify support/resistance levels
  - Detect trend strength and direction
  - Predict potential breakouts/breakdowns

### 4. nextgen_models/nextgen_sentiment_analysis/sentiment_analysis_model.py
- **Purpose**: Comprehensive financial text processing and analysis
- **Functions**:
  - Process news articles and social media
  - Score sentiment on financial texts
  - Detect sentiment shifts and trends
  - Identify key sentiment drivers
  - Correlate sentiment with price action
  - Extract structured data from financial documents
  - Process earnings reports and SEC filings

### 5. nextgen_models/nextgen_context_model/context_model.py
- **Purpose**: Retrieval-augmented generation for financial context
- **Functions**:
  - Access knowledge bases for context
  - Incorporate historical patterns and precedents
  - Provide factual grounding for decisions
  - Integrate domain knowledge into analysis
  - Connect current conditions with historical context

### 6. nextgen_models/nextgen_trader/trade_model.py
- **Purpose**: Intelligent trade execution
- **Functions**:
  - Determine optimal execution strategy
  - Implement dynamic position sizing
  - Manage execution timing
  - Minimize market impact
  - Connect with Alpaca MCP for execution
  - Monitor execution quality
  - Track positions and order status
  - Implement risk management logic

### 7. nextgen_models/nextgen_risk_assessment/risk_assessment_model.py
- **Purpose**: Assess and manage risk at position and portfolio levels
- **Functions**:
  - Calculate position-level risk metrics
  - Evaluate portfolio risk exposure
  - Implement risk limits and checks
  - Monitor drawdown and volatility
  - Recommend position sizing based on risk

### 8. nextgen_models/nextgen_decision/decision_model.py
- **Purpose**: Make final trading decisions
- **Functions**:
  - Integrate analysis from all components
  - Apply risk management constraints
  - Determine trade direction and confidence
  - Prioritize opportunities
  - Implement decision rules
  - Generate final trade decisions

### 9. nextgen_models/nextgen_fundamental_analysis/fundamental_analysis_model.py
- **Purpose**: Analyze company fundamentals
- **Functions**:
  - Evaluate financial statements
  - Calculate financial ratios
  - Assess growth metrics
  - Compare sector performance
  - Analyze earnings and guidance

## MCP Tools and Data Integration

### 10. mcp_tools/data_mcp/base_data_mcp.py
- **Purpose**: Base class for data MCP integrations
- **Functions**:
  - Provide common interface for data sources
  - Handle connection management
  - Implement caching and rate limiting
  - Define tool registration patterns

### 11. mcp_tools/data_mcp/polygon_rest_mcp.py
- **Purpose**: Interface with Polygon.io REST API
- **Functions**:
  - Fetch historical market data
  - Retrieve reference data (splits, dividends)
  - Get fundamental company data
  - Access market indices and indicators
  - Handle API rate limiting and pagination

### 12. mcp_tools/data_mcp/polygon_ws_mcp.py
- **Purpose**: Interface with Polygon.io WebSocket API
- **Functions**:
  - Stream real-time market data
  - Process live trades and quotes
  - Maintain persistent connections
  - Handle reconnection logic
  - Buffer and process streaming data

### 13. mcp_tools/data_mcp/yahoo_finance_mcp.py
- **Purpose**: Interface with Yahoo Finance
- **Functions**:
  - Access fundamental company data
  - Retrieve historical price data
  - Get options chains information
  - Access financial statements
  - Fetch analyst recommendations

### 14. mcp_tools/data_mcp/reddit_mcp.py
- **Purpose**: Interface with Reddit for sentiment data
- **Functions**:
  - Monitor financial subreddits
  - Track trending tickers
  - Gather sentiment from user posts
  - Identify emerging narratives
  - Detect unusual activity spikes

### 15. mcp_tools/data_mcp/unusual_whales_mcp.py
- **Purpose**: Interface with Unusual Whales API
- **Functions**:
  - Track unusual options activity
  - Monitor large block trades
  - Detect potential informed trading
  - Identify unusual volume patterns
  - Track institutional money flow

### 16. mcp_tools/alpaca_mcp/alpaca_mcp.py
- **Purpose**: Interface with Alpaca trading platform
- **Functions**:
  - Execute buy/sell orders
  - Manage positions and orders
  - Stream market data
  - Access account information
  - Handle authentication and API limits
  - Process execution reports

### 17. mcp_tools/db_mcp/redis_mcp.py
- **Purpose**: Interface with Redis for caching and messaging
- **Functions**:
  - Cache frequently used data
  - Implement pub/sub for event distribution
  - Store temporary state information
  - Maintain distributed state
  - Implement distributed locking
  - Enable fast data access

## Analysis MCP Tools

### 18. mcp_tools/analysis_mcp/peak_detection_mcp.py
- **Purpose**: Detect peaks and troughs in price series
- **Functions**:
  - Identify local maxima and minima
  - Calculate prominence and width
  - Filter based on significance
  - Detect potential reversal points

### 19. mcp_tools/analysis_mcp/drift_detection_mcp.py
- **Purpose**: Detect statistical drift in price or other metrics
- **Functions**:
  - Identify regime changes
  - Detect distribution shifts
  - Signal potential trend changes
  - Monitor for statistical anomalies

### 20. mcp_tools/analysis_mcp/slippage_analysis_mcp.py
- **Purpose**: Analyze execution quality and slippage
- **Functions**:
  - Calculate price impact of trades
  - Compare execution price to arrival price
  - Measure implementation shortfall
  - Recommend optimal execution strategies

### 21. mcp_tools/analysis_mcp/technical_indicators_mcp.py
- **Purpose**: Calculate technical indicators
- **Functions**:
  - Compute moving averages, oscillators, and trend indicators
  - Generate signals based on technical patterns
  - Provide relative strength measurements
  - Calculate volatility metrics

## Infrastructure Components

### 22. local_redis/redis_server.py
- **Purpose**: Manage local Redis server
- **Functions**:
  - Start and configure Redis instance
  - Set up persistence
  - Configure memory limits
  - Manage connections

### 23. local_vectordb/vectordb_server.py
- **Purpose**: Manage local vector database
- **Functions**:
  - Store and retrieve embeddings
  - Enable similarity search
  - Manage vector collections
  - Handle persistence

### 24. monitoring/system_monitor.py
- **Purpose**: Monitor system health and performance
- **Functions**:
  - Track resource utilization
  - Collect performance metrics
  - Log system events
  - Send alerts for critical issues
  - Provide performance visualizations

## Configuration and Environment

### 25. .env and .env.example
- **Purpose**: Store environment configuration
- **Functions**:
  - Manage API keys and credentials
  - Configure service endpoints
  - Set operational parameters
  - Specify environment-specific settings
  - Control debugging options

## System Workflow

The system operates through an event-driven architecture with the following key workflow:

1. **Stock Discovery**: The selection component identifies potential trading candidates through multi-tier filtering
2. **Data Gathering**: MCP data connectors collect relevant information for the selected stocks
3. **Analysis**: Multiple analysis components process the gathered data:
   - Sentiment Analysis analyzes news, social media, and financial documents
   - Market Analysis generates price predictions and identifies patterns
   - Context Model provides historical context and precedents
   - Fundamental Analysis evaluates company financials
4. **Risk Assessment**: The risk model evaluates position and portfolio-level risk
5. **Decision Making**: The decision model integrates all analysis and makes trading decisions
6. **Execution**: The trade model implements trades via Alpaca MCP
7. **Monitoring**: The system monitor tracks performance and system health

## Integration Points

- **Redis** serves as the primary message bus and caching layer
- **Vector Database** provides efficient storage and retrieval of embeddings
- **MCP Tools** provide standardized interfaces to external services
- **AutoGen Orchestrator** coordinates specialized agents

## Dependencies

- **AutoGen Framework**: Microsoft's framework for agent orchestration
- **LLM Models**: Large language models accessed via OpenRouter
- **Market Data Providers**: Polygon.io, Yahoo Finance, Reddit, Unusual Whales
- **Execution Broker**: Alpaca
- **Infrastructure**: Redis, Vector Database, Monitoring

## Development and Deployment

The system is designed to run as a collection of services, each handling a specific part of the trading workflow. Components communicate through a well-defined event system, allowing for distributed deployment and scaling of individual components as needed.
