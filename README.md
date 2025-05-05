# NextGen AI Trading System

A comprehensive, AI-powered trading system built on a modular architecture with Model Context Protocol (MCP) for tool integration and Microsoft AutoGen for agent orchestration.

## System Overview

The NextGen AI Trading System is an end-to-end solution that combines traditional market data analysis with cutting-edge LLM-based financial intelligence. The system integrates multiple data sources via MCP tools, applies sophisticated analysis using specialized financial language models, and executes trades through the Alpaca trading platform.

For detailed information about the system:
- See [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) for a complete description of all components
- See [docs/SYSTEM_FLOW.md](docs/SYSTEM_FLOW.md) for data flow and interaction diagrams

## Directory Structure

```
nextgen/
├── .env                      # Environment variables for APIs and configurations
├── .env.example              # Example environment file
├── config/                   # Configuration files
│   ├── analysis_mcp/         # Analysis MCP tool configurations
│   ├── nextgen_decision/     # Decision model configurations
│   ├── nextgen_risk/         # Risk model configurations
│   ├── nextgen_select/       # Selection model configurations
│   └── nextgen_trader/       # Trader model configurations
├── docs/                     # Documentation
│   ├── ENVIRONMENT_VARIABLES.md # Details on required environment variables
│   ├── SYSTEM_ARCHITECTURE.md # Detailed component architecture
│   ├── SYSTEM_FLOW.md        # System flow and data diagram
│   └── SYSTEM_HARDWARE.md    # Hardware requirements and setup
├── local_redis/              # Local Redis server management
│   ├── README.md             # Redis setup instructions
│   ├── redis_server.py       # Redis server implementation
│   └── start_redis_server.py # Redis server startup script
├── local_vectordb/           # Local vector database management
│   ├── README.md             # VectorDB setup instructions
│   ├── vectordb_server.py    # VectorDB server implementation
│   └── start_vectordb_server.py # VectorDB server startup script
├── mcp_tools/                # MCP tool implementations
│   ├── alpaca_mcp/           # Alpaca trading API tools
│   │   └── alpaca_mcp.py     # Alpaca MCP implementation
│   ├── base_mcp_server.py    # Base MCP server implementation
│   ├── document_analysis_mcp/ # Consolidated document analysis tools
│   │   └── document_analysis_mcp.py # Document analysis implementation
│   ├── financial_data_mcp/   # Consolidated financial data tools
│   │   └── financial_data_mcp.py # Financial data implementation
│   ├── financial_text_mcp/   # Consolidated financial text analysis
│   │   └── financial_text_mcp.py # Financial text analysis implementation
│   ├── risk_analysis_mcp/    # Consolidated risk analysis tools
│   │   └── risk_analysis_mcp.py # Risk analysis implementation
│   ├── time_series_mcp/      # Consolidated time series analysis
│   │   └── time_series_mcp.py # Time series analysis implementation
│   ├── trading_mcp/          # Consolidated trading operations
│   │   └── trading_mcp.py    # Trading operations implementation
│   └── vector_store_mcp/     # Vector database tools (ChromaDB)
│       └── vector_store_mcp.py # Vector store implementation
├── monitoring/               # System monitoring tools
│   ├── README.md             # Monitoring setup instructions
│   ├── system_monitor.py     # System monitoring implementation
│   └── monitoring_utils.py   # Monitoring utilities
├── nextgen_models/           # Core model implementations
│   ├── autogen_orchestrator/ # AutoGen orchestration
│   │   └── autogen_model.py  # AutoGen implementation
│   ├── nextgen_context_model/ # Context and knowledge models
│   │   └── context_model.py  # Context model implementation
│   ├── nextgen_decision/     # Decision models
│   │   └── decision_model.py # Decision model implementation
│   ├── nextgen_fundamental_analysis/ # Fundamental analysis
│   │   └── fundamental_analysis_model.py # Fundamental analysis implementation
│   ├── nextgen_market_analysis/ # Market analysis
│   │   └── market_analysis_model.py # Market analysis implementation
│   ├── nextgen_risk_assessment/ # Risk assessment
│   │   └── risk_assessment_model.py # Risk assessment implementation
│   ├── nextgen_select/       # Stock selection
│   │   └── select_model.py   # Selection model implementation
│   ├── nextgen_sentiment_analysis/ # Sentiment analysis
│   │   └── sentiment_analysis_model.py # Sentiment analysis implementation
│   └── nextgen_trader/       # Trade execution
│       └── trade_model.py    # Trade model implementation
└── tests/                    # Test suite
    ├── mcp_tools/            # MCP tool tests
    │   ├── alpaca_mcp/       # Alpaca MCP tests
    │   ├── vector_store_mcp/ # Vector store tests
    │   └── financial_data_mcp/ # Financial data tests
    └── nextgen_models/       # Model tests
        └── nextgen_trader/   # Trader model tests
```

## Key Components

### Core Trading Models

- **AutoGen Orchestrator**: Central orchestration using Microsoft's AutoGen framework for agent coordination
- **Selection Model**: Identifies potential trading candidates through multi-tier filtering
- **Decision Model**: Integrates analysis from all components to make trading decisions
- **Trade Model**: Implements dynamic position sizing and optimal execution strategies
- **Risk Assessment**: Evaluates position and portfolio risk

### Analysis Models

- **Sentiment Analysis**: Processes news and social media for sentiment analysis
- **Market Analysis**: Generates price predictions and identifies technical patterns
- **Context Model**: Provides historical context and precedents using vector retrieval
- **Fundamental Analysis**: Evaluates company financials and metrics

### MCP Tools

- **Financial Data MCP**: Real-time and historical market data consolidated from Polygon.io, Yahoo Finance, and other sources
- **Document Analysis MCP**: Consolidated document processing, embeddings generation, query reformulation, and relevance feedback
- **Risk Analysis MCP**: Consolidated risk evaluation, portfolio optimization, and drift detection tools
- **Time Series MCP**: Consolidated time series analysis, technical indicators, and forecasting
- **Trading MCP**: Consolidated order execution and position management
- **Vector Store MCP**: ChromaDB integration for vector storage and similarity search

### Infrastructure

- **Redis**: Event bus and caching layer
- **ChromaDB**: Vector database for embeddings and efficient similarity search
- **System Monitor**: Performance tracking and centralized logging

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in required API keys and configuration
3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Redis:
   ```bash
   python local_redis/start_redis_server.py
   ```
5. Start Vector Database:
   ```bash
   python local_vectordb/start_vectordb_server.py
   ```

## Usage

1. Launch the AutoGen orchestrator:
   ```bash
   python -m nextgen_models.autogen_orchestrator.autogen_model
   ```

2. Monitor performance through the system monitoring tools

## Model Integration

The system uses LLMs via OpenRouter:

- Anthropic Claude 3 Opus for in-depth financial analysis
- Meta Llama 3 70B for complex financial reasoning
- Google Gemini Pro for additional analysis capabilities

See [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) for detailed model sources and integration.

## MCP Tool Integration

The Model Context Protocol (MCP) provides a standardized interface for models to access external tools and resources:

1. **Consolidated Data Tools**: MCP tools for market data from multiple providers consolidated into logical groups
2. **Consolidated Analysis Tools**: MCP tools for specialized financial analysis grouped by function
3. **Vector Store Tools**: ChromaDB integration for efficient semantic search and RAG functionality
4. **Execution**: MCP tools for trade execution and order management
5. **Infrastructure**: MCP tools for database access and coordination

This architecture allows for easy extension with new data sources and analysis capabilities without changing core component code.

## AutoGen Agent Architecture

The system uses Microsoft's AutoGen framework for agent orchestration:

- **Specialized Agents**: Each agent handles a specific domain (selection, analysis, execution, etc.)
- **Tool Registration**: MCP tools are registered with appropriate agents
- **Group Chat**: Agents collaborate in group chats to solve complex problems
- **LLM Integration**: LLMs power agent reasoning via OpenRouter

See [docs/SYSTEM_FLOW.md](docs/SYSTEM_FLOW.md) for details on agent interactions and communication.

## Data Flow

The system operates through an event-driven architecture with the following key workflow:

1. **Stock Discovery**: The selection model identifies potential trading candidates
2. **Data Gathering**: MCP data tools collect relevant information for the selected stocks
3. **Analysis**: Multiple analysis models process the gathered data
4. **Decision Making**: The decision model integrates all analysis and makes trading decisions
5. **Execution**: The trade model implements trades via Alpaca MCP
6. **Position Management**: The trade model tracks and manages active positions
7. **Performance Analysis**: The monitoring components measure strategy effectiveness

See [docs/SYSTEM_FLOW.md](docs/SYSTEM_FLOW.md) for a complete visualization of the data flow.

## License

This project is released under the [MIT License](LICENSE).

## References

- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [Alpaca Trading API](https://alpaca.markets/)
- [Polygon.io API](https://polygon.io/)
- [OpenRouter API](https://openrouter.ai/)
- [ChromaDB](https://www.trychroma.com/)
