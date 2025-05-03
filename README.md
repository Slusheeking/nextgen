# FinGPT AI Day Trading System

A comprehensive, AI-powered day trading system built on the FinGPT framework, using large language models to analyze financial markets, select stocks, generate trading signals, and execute trades.

## System Overview

The FinGPT Day Trading System is an end-to-end solution that combines traditional market data analysis with cutting-edge LLM-based financial intelligence. The system integrates multiple data sources, applies sophisticated analysis using specialized financial language models, and executes trades through the Alpaca trading platform.

For detailed information about the system:
- See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for a complete description of all components
- See [SYSTEM_FLOW.md](SYSTEM_FLOW.md) for data flow and interaction diagrams

## Directory Structure

```
nextgen/
├── .env                      # Environment variables for APIs and configurations
├── .env.example              # Example environment file
├── README.md                 # This file - project documentation
├── requirements.txt          # Consolidated project dependencies
├── SYSTEM_ARCHITECTURE.md    # Detailed component architecture
├── SYSTEM_FLOW.md            # System flow and data diagram
├── alpaca/                   # Alpaca trading integration
│   └── alpaca_api.py         # Alpaca API client implementation
├── data/                     # Data sources and preprocessing
│   ├── data_preprocessor.py  # Data normalization and feature engineering
│   ├── polygon_rest.py       # Polygon.io REST API client
│   ├── polygon_ws.py         # Polygon.io WebSocket API client
│   ├── reddit.py             # Reddit API client for sentiment data
│   ├── unusual_whales.py     # Unusual Whales API client for options flow
│   └── yahoo_finance.py      # Yahoo Finance API client
├── fingpt/                   # Core FinGPT components
│   ├── fingpt_bench/         # Benchmarking and performance evaluation
│   │   └── bench_model.py    # Benchmark model implementation
│   ├── fingpt_execution/     # Trade execution component
│   │   └── execution_model.py # Execution strategy implementation
│   ├── fingpt_finnlp/        # Financial NLP and sentiment analysis
│   │   └── finnlp_model.py   # Financial NLP model implementation
│   ├── fingpt_forcaster/     # Price and trend forecasting
│   │   └── forcaster_model.py # Forecasting model implementation
│   ├── fingpt_lora/          # LoRA adaptation for fine-tuning LLMs
│   │   └── lora_model.py     # LoRA model implementation
│   ├── fingpt_orchestrator/  # Central system orchestrator
│   │   └── orchestrator_model.py # Orchestrator implementation
│   ├── fingpt_order/         # Order management system
│   │   └── order_managment_model.py # Order management implementation
│   ├── fingpt_rag/           # Retrieval-augmented generation
│   │   └── rag_model.py      # RAG model implementation
│   └── fingpt_selection/     # Stock selection and filtering
│       └── selection_model.py # Stock selection implementation
├── influxdb/                 # Time series database integration
│   └── influxdb_manager.py   # InfluxDB client and manager
├── loki/                     # Logging infrastructure
│   └── loki_manager.py       # Loki logging client
├── prometheus/               # Metrics and monitoring
│   └── prometheus_manager.py # Prometheus metrics manager
├── redis/                    # Caching and messaging
│   └── redis_manager.py      # Redis client and manager
└── systemd/                  # System deployment configuration
```

## Key Components

### Core Trading Engine

- **Orchestrator**: Central brain of the system that coordinates all components and makes trading decisions
- **Selection**: Identifies potential trading candidates through multi-tier filtering
- **Execution**: Implements dynamic position sizing and optimal execution strategies
- **Order Management**: Tracks open positions and implements risk management

### Analysis Components

- **FinNLP**: Comprehensive financial text processing and sentiment analysis
- **Forecaster**: Time series forecasting and technical pattern recognition
- **RAG**: Retrieval-augmented generation for incorporating financial knowledge

### Data Sources

- **Market Data**: Real-time and historical price data from Polygon.io and Yahoo Finance
- **Alternative Data**: Social sentiment from Reddit, options flow from Unusual Whales
- **Fundamental Data**: Financial statements, earnings reports, and SEC filings

### Infrastructure

- **Redis**: Event bus and caching layer
- **InfluxDB**: Time series storage for market data and performance metrics
- **Prometheus & Loki**: Monitoring and logging infrastructure

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in required API keys and configuration
3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install infrastructure components (Redis, InfluxDB, Prometheus, Loki) following their respective documentation

## Usage

1. Start infrastructure services:
   ```bash
   systemctl start redis influxdb prometheus loki
   ```

2. Launch the FinGPT orchestrator:
   ```bash
   python -m fingpt.fingpt_orchestrator.orchestrator_model
   ```

3. Monitor system performance through the Prometheus dashboard

## Model Integration

The system uses several pretrained financial models:

- FinGPT models from HuggingFace (FinGPT-v3-llama2-7b, FinGPT-RAG, FinGPT-Forecaster)
- OpenAI API for complex financial analysis
- Anthropic Claude for document analysis
- Mistral AI for additional analysis capabilities

See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for detailed model sources and URLs.

## Data Flow

The system operates through an event-driven architecture with the following key workflow:

1. **Stock Discovery**: The selection component identifies potential trading candidates
2. **Data Gathering**: Data connectors collect relevant information for the selected stocks
3. **Analysis**: Multiple analysis components process the gathered data
4. **Decision Making**: The orchestrator integrates all analysis and makes trading decisions
5. **Execution**: The execution component implements trades with dynamic sizing
6. **Position Management**: The order management component tracks and manages active positions
7. **Performance Analysis**: The benchmarking component measures strategy effectiveness

See [SYSTEM_FLOW.md](SYSTEM_FLOW.md) for a complete visualization of the data flow.

## License

This project is based on the FinGPT framework by AI4Finance Foundation and is released under the [MIT License](LICENSE).

## References

- [FinGPT GitHub Repository](https://github.com/AI4Finance-Foundation/FinGPT)
- [Alpaca Trading API](https://alpaca.markets/)
- [Polygon.io API](https://polygon.io/)
