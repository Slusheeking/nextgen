# FinGPT AI Day Trading System Flow

This document visualizes the complete system flow of the FinGPT AI Day Trading System, showing how data and control flow through the various components.

## Complete System Flow Diagram

```mermaid
graph TD
    %% Define external data sources
    DS1[Polygon REST API] -.-> DR1
    DS2[Polygon WS API] -.-> DR2
    DS3[Yahoo Finance] -.-> DR3
    DS4[Reddit API] -.-> DR4
    DS5[Unusual Whales] -.-> DR5
    DS6[SEC EDGAR] -.-> DR6
    DS7[FRED] -.-> DR7
    DS8[News APIs] -.-> DR8
    
    %% Data Connectors
    subgraph "Data Connectors"
        DR1[polygon_rest.py]
        DR2[polygon_ws.py]
        DR3[yahoo_finance.py]
        DR4[reddit.py]
        DR5[unusual_whales.py]
        DR6[External - SEC API]
        DR7[External - FRED API]
        DR8[External - News APIs]
    end
    
    %% Stock Selection
    subgraph "Stock Selection"
        SS1[fingpt_selection]
    end
    
    %% Data Processing
    subgraph "Data Processing"
        DP1[data_preprocessor.py]
    end
    
    %% Analysis Components
    subgraph "Analysis Components"
        AC1[fingpt_finnlp]
        AC2[fingpt_forcaster]
        AC3[fingpt_rag]
    end
    
    %% Orchestration
    subgraph "Orchestration"
        OR1[fingpt_orchestrator]
    end
    
    %% Trading Components
    subgraph "Trading Components"
        TR1[fingpt_execution]
        TR2[fingpt_order]
    end
    
    %% External Execution
    EX1[Alpaca API] -.-> TR1
    
    %% Infrastructure
    subgraph "Infrastructure"
        IN1[(Redis)]
        IN2[(InfluxDB)]
        IN3[Prometheus]
        IN4[Loki]
    end
    
    %% Support Components
    subgraph "Support Components"
        SC1[fingpt_bench]
        SC2[fingpt_lora]
    end
    
    %% Model Providers
    subgraph "Model Providers"
        MP1["FinGPT Models (HuggingFace)"]
        MP2["OpenAI API"]
        MP3["Anthropic Claude API"]
        MP4["Mistral AI"]
    end
    
    %% Flow connections
    DR1 --> DP1
    DR2 --> DP1
    DR3 --> DP1
    DR4 --> DP1
    DR5 --> DP1
    DR6 --> DP1
    DR7 --> DP1
    DR8 --> DP1
    
    DR1 --> SS1
    DR2 --> SS1
    DR3 --> SS1
    
    SS1 --> DP1
    
    DP1 --> AC1
    DP1 --> AC2
    DP1 --> AC3
    
    AC1 --> OR1
    AC2 --> OR1
    AC3 --> OR1
    SS1 --> OR1
    
    OR1 --> TR1
    OR1 --> TR2
    
    TR1 --> EX1
    TR2 --> EX1
    
    OR1 <--> IN1
    OR1 --> IN2
    OR1 --> IN3
    OR1 --> IN4
    
    SC1 <--> OR1
    SC2 <--> OR1
    
    MP1 -.-> AC1
    MP1 -.-> AC2
    MP1 -.-> AC3
    MP1 -.-> OR1
    
    MP2 -.-> AC1
    MP2 -.-> AC3
    MP2 -.-> OR1
    
    MP3 -.-> AC1
    
    MP4 -.-> AC1
    MP4 -.-> AC3
    
    %% Event Flow
    IN1 ~~~ OR1
    
    %% Style definitions
    classDef dataSource fill:#90CAF9,stroke:#0D47A1,stroke-width:1px;
    classDef dataConnector fill:#81D4FA,stroke:#01579B,stroke-width:1px;
    classDef processing fill:#80CBC4,stroke:#004D40,stroke-width:1px;
    classDef analysis fill:#A5D6A7,stroke:#1B5E20,stroke-width:1px;
    classDef orchestration fill:#FFE082,stroke:#FF6F00,stroke-width:2px;
    classDef trading fill:#FFAB91,stroke:#BF360C,stroke-width:1px;
    classDef infrastructure fill:#CE93D8,stroke:#4A148C,stroke-width:1px;
    classDef support fill:#BCAAA4,stroke:#3E2723,stroke-width:1px;
    classDef models fill:#F48FB1,stroke:#880E4F,stroke-width:1px;
    
    %% Apply styles
    class DS1,DS2,DS3,DS4,DS5,DS6,DS7,DS8 dataSource;
    class DR1,DR2,DR3,DR4,DR5,DR6,DR7,DR8 dataConnector;
    class DP1 processing;
    class SS1 processing;
    class AC1,AC2,AC3 analysis;
    class OR1 orchestration;
    class TR1,TR2 trading;
    class EX1 trading;
    class IN1,IN2,IN3,IN4 infrastructure;
    class SC1,SC2 support;
    class MP1,MP2,MP3,MP4 models;
```

## Event Flow Sequence

The system operates through the following event flow sequence:

1. **Market Data Collection**
   - Data connectors retrieve data from external sources
   - Real-time and historical data streams are processed

2. **Stock Universe Generation and Filtering**
   - `fingpt_selection` creates the initial trading universe
   - Applies multi-tier filtering with both rules and LLM intelligence
   - Selected candidates are passed to data preprocessor

3. **Data Preprocessing**
   - `data_preprocessor.py` normalizes and prepares data for analysis
   - Creates feature vectors and technical indicators
   - Formats data for model consumption

4. **Multi-faceted Analysis**
   - `fingpt_finnlp` analyzes financial texts and sentiment
   - `fingpt_forcaster` generates price predictions
   - `fingpt_rag` provides historical context and precedents

5. **Decision Orchestration**
   - `fingpt_orchestrator` integrates all analysis results
   - Applies portfolio constraints and risk management rules
   - Makes final trading decisions

6. **Trade Execution and Management**
   - `fingpt_execution` determines position sizing and execution strategy
   - `fingpt_order` manages order lifecycle and monitors positions
   - Trades are submitted to Alpaca for execution

7. **Performance Analysis and Adaptation**
   - `fingpt_bench` evaluates trading performance
   - `fingpt_lora` adapts models based on performance feedback
   - System parameters are adjusted based on market conditions

## Event Bus Architecture

The Redis message bus enables event-driven communication between components:

- **Market Events**: Price changes, economic announcements, news events
- **System Events**: Component status, error conditions, performance metrics
- **Trading Events**: Order execution, position changes, P&L updates
- **Analysis Events**: New sentiment scores, prediction updates, pattern detections

Each component both produces and consumes events according to its role in the system, creating a responsive and adaptable trading architecture.

## Data Flow Rates

| Component Connection | Typical Data Rate | Update Frequency |
|----------------------|-------------------|------------------|
| Market Data → System | 5-50 MB/min | Real-time (ms) to 15-min |
| Selection → Preprocessor | 100-500 KB/event | 15-60 min |
| Analysis → Orchestrator | 10-50 KB/event | 1-15 min |
| Orchestrator → Execution | 1-5 KB/event | 1-15 min |
| System → Monitoring | 100-500 KB/min | Continuous |
