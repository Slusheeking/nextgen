# NextGen AI Trading System

A comprehensive AI-powered trading system orchestrated by language models, built on a modular architecture using MCP (Model Context Protocol) for efficient tool integration and AutoGen for agent orchestration.

## System Architecture

The NextGen AI Trading System is organized as a modular, event-driven architecture with several key components:

1. **Data Ingestion Layer**: Collects market data from multiple sources through consolidated MCP data tools
2. **Stock Selection Layer**: Identifies trading candidates through multi-tier filtering
3. **Analysis Layer**: Processes data through specialized components and consolidated MCP analysis tools
4. **Decision Layer**: Orchestrates analysis results into actionable trading decisions
5. **Execution Layer**: Implements trades with dynamic position sizing via Trading MCP
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
  - Options data and market indicators via Financial Data MCP
- **Yahoo Finance**: https://finance.yahoo.com/
  - Earnings reports, analyst recommendations, news via Financial Data MCP
- **Unusual Whales**: https://unusualwhales.com/
  - Options flow and unusual activity detection via Financial Data MCP
- **Reddit**: https://www.reddit.com/
  - Social sentiment and retail investor activity via Financial Data MCP

### Trading Integration
- **Alpaca Markets**: https://alpaca.markets/
  - Order execution and position management via Trading MCP

### Data Storage
- **Redis**: In-memory data store for real-time data and coordination
- **ChromaDB**: Vector database for embeddings storage
  - Integrated via Vector Store MCP for efficient similarity search and RAG functionality

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
  - Process and store documents with embeddings in ChromaDB
  - Retrieve relevant documents via semantic search
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
  - Connect with Trading MCP for execution
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
  - Retrieve context information from ChromaDB via Context Model
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

## Consolidated MCP Tools

### 10. mcp_tools/base_mcp_server.py
- **Purpose**: Base class for MCP integrations
- **Functions**:
  - Provide common interface for all MCP tools
  - Handle connection management
  - Implement caching and rate limiting
  - Define tool registration patterns

### 11. mcp_tools/financial_data_mcp/financial_data_mcp.py
- **Purpose**: Consolidated interface for financial market data
- **Functions**:
  - Fetch historical and real-time market data from multiple sources
  - Retrieve reference data (splits, dividends)
  - Get fundamental company data and financial statements
  - Access market indices and indicators
  - Handle API rate limiting and pagination
  - Stream real-time data
  - Track social media sentiment and options flow

### 12. mcp_tools/document_analysis_mcp/document_analysis_mcp.py
- **Purpose**: Consolidated document processing and analysis
- **Functions**:
  - Process documents (clean, extract metadata, chunk)
  - Generate embeddings for text
  - Deduplicate chunks
  - Reformulate queries for improved retrieval
  - Incorporate relevance feedback
  - Rerank search results based on feedback

### 13. mcp_tools/risk_analysis_mcp/risk_analysis_mcp.py
- **Purpose**: Consolidated risk analysis tools
- **Functions**:
  - Calculate position-level risk metrics
  - Optimize portfolio composition
  - Detect market regime changes and distribution shifts
  - Generate risk scenarios
  - Analyze risk attribution
  - Calculate confidence scores
  - Calculate optimal position sizing

### 14. mcp_tools/time_series_mcp/time_series_mcp.py
- **Purpose**: Consolidated time series analysis
- **Functions**:
  - Calculate technical indicators
  - Detect peaks and troughs in price series
  - Identify statistical patterns
  - Analyze correlations between assets
  - Generate forecasts

### 15. mcp_tools/trading_mcp/trading_mcp.py
- **Purpose**: Consolidated trading operations
- **Functions**:
  - Execute buy/sell orders
  - Manage positions and orders
  - Stream market data
  - Access account information
  - Handle authentication and API limits
  - Process execution reports
  - Analyze slippage and execution quality

### 16. mcp_tools/vector_store_mcp/vector_store_mcp.py
- **Purpose**: Interface with ChromaDB vector database
- **Functions**:
  - Store document embeddings and metadata
  - Perform similarity search on embeddings
  - Manage collections of vectors
  - Handle persistence and retrieval
  - Support filtering based on metadata

## Infrastructure Components

### 17. local_redis/redis_server.py
- **Purpose**: Manage local Redis server
- **Functions**:
  - Start and configure Redis instance
  - Set up persistence
  - Configure memory limits
  - Manage connections

### 18. local_vectordb/vectordb_server.py
- **Purpose**: Manage local vector database
- **Functions**:
  - Store and retrieve embeddings
  - Enable similarity search
  - Manage vector collections
  - Handle persistence

### 19. monitoring/system_monitor.py
- **Purpose**: Monitor system health and performance
- **Functions**:
  - Track resource utilization
  - Collect performance metrics
  - Log system events
  - Send alerts for critical issues
  - Provide performance visualizations

## Configuration and Environment

### 20. .env and .env.example
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
2. **Data Gathering**: Consolidated MCP data connectors collect relevant information for the selected stocks
3. **Analysis**: Multiple analysis components process the gathered data:
   - Sentiment Analysis analyzes news, social media, and financial documents
   - Market Analysis generates price predictions and identifies patterns
   - Context Model provides historical context from ChromaDB vector store
   - Fundamental Analysis evaluates company financials
4. **Risk Assessment**: The risk model evaluates position and portfolio-level risk
5. **Decision Making**: The decision model integrates all analysis and makes trading decisions
6. **Execution**: The trade model implements trades via Trading MCP
7. **Monitoring**: The system monitor tracks performance and system health

## Integration Points

- **Redis** serves as the primary message bus and caching layer
- **ChromaDB** provides efficient storage and retrieval of embeddings via Vector Store MCP
- **MCP Tools** provide standardized consolidated interfaces to external services
- **AutoGen Orchestrator** coordinates specialized agents

## Dependencies

- **AutoGen Framework**: Microsoft's framework for agent orchestration
- **LLM Models**: Large language models accessed via OpenRouter
- **Market Data Providers**: Polygon.io, Yahoo Finance, Reddit, Unusual Whales
- **Execution Broker**: Alpaca
- **Infrastructure**: Redis, ChromaDB, Monitoring

## Development and Deployment

The system is designed to run as a collection of services, each handling a specific part of the trading workflow. Components communicate through a well-defined event system, allowing for distributed deployment and scaling of individual components as needed.
