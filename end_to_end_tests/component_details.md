# NextGen Component Details

This document provides detailed descriptions of the key models and MCP (Model Context Protocol) tools within the NextGen trading system.

## Models

### `autogen_model`

*   **Purpose and Functionality:** The `autogen_model` acts as the central orchestrator for the NextGen Models system. It utilizes Microsoft's AutoGen framework to coordinate various specialized agents (Selection, Data, FinNLP, Forecaster, RAG, Execution, Monitoring) to perform financial analysis and trading decisions. It manages the overall trading workflow, initiates communication between agents, and processes the final decisions.
*   **MCP Tool Interactions:** The `autogen_model` interacts with several MCP tools indirectly through the functions registered with its `UserProxyAgent`. These registered functions, implemented within the `AutoGenOrchestrator` class, call methods on instances of the other NextGen models (SelectionModel, SentimentAnalysisModel, etc.). These model instances, in turn, interact directly with the underlying MCP tools (TradingMCP, FinancialDataMCP, RedisMCP, etc.). The orchestrator itself also has registered functions like `use_mcp_tool` and `list_mcp_tools` that allow agents to directly interact with connected MCP servers.

### `context_model`

*   **Purpose and Functionality:** The `context_model` is responsible for gathering, processing, and managing contextual information from various sources to support other NextGen models, particularly for RAG (Retrieval Augmented Generation) purposes. It integrates with data retrieval, document analysis, and vector storage MCP tools to fetch data, process documents, generate embeddings, store/retrieve vectors, and incorporate relevance feedback.
*   **MCP Tool Interactions:** The `context_model` directly interacts with the `DocumentAnalysisMCP` (for processing documents, generating embeddings, query reformulation, and relevance feedback) and the `VectorStoreMCP` (for adding and searching documents in the vector database). It also interacts with various data source MCPs (like Polygon News, Reddit, Yahoo Finance, Yahoo News, Unusual Whales) via a generic `fetch_data` method and specific `use_*_tool` methods registered with its AutoGen agents. It uses the `RedisMCP` for storing and retrieving general contextual data and potentially for managing feedback streams.

### `decision_model`

*   **Purpose and Functionality:** The `decision_model` is the core decision-making component of the NextGen trading system. It aggregates analysis results from all other models (Selection, Sentiment, Market, Fundamental, Risk), considers portfolio constraints and market conditions, applies risk management rules, and makes final trading decisions (buy, sell, hold) with associated confidence levels and reasoning.
*   **MCP Tool Interactions:** The `decision_model` directly interacts with the `RiskAnalysisMCP` (for decision analytics, portfolio optimization, and drift detection), the `FinancialDataMCP` (for market data needed for market state evaluation), and the `RedisMCP` (for inter-model communication, storing decisions, and retrieving data from other models). It also interacts with the `ContextModel` for RAG functionality. It receives analysis reports from the SentimentAnalysisModel, FundamentalAnalysisModel, and MarketAnalysisModel by reading from their respective Redis streams/keys and sends actionable decisions to the TradeModel.

### `fundamental_analysis_model`

*   **Purpose and Functionality:** The `fundamental_analysis_model` is responsible for analyzing the financial health and intrinsic value of companies. It retrieves financial statements, calculates key ratios, evaluates growth trends, processes earnings reports, and compares companies to their peers and sectors. It provides fundamental insights to the Decision Model.
*   **MCP Tool Interactions:** The `fundamental_analysis_model` directly interacts with the `FinancialDataMCP` (for retrieving financial statements, market data, and earnings reports), the `RiskAnalysisMCP` (for calculating financial ratios, scoring financial health, analyzing growth, and calculating value metrics), and the `RedisMCP` (for caching fundamental data and sending feedback/analysis reports to other models via streams). It sends feedback to the SelectionModel and analysis reports to the DecisionModel.

### `market_analysis_model`

*   **Purpose and Functionality:** The `market_analysis_model` focuses on technical analysis of market data. It calculates technical indicators, detects chart patterns, identifies support and resistance levels, and performs market scanning to find potential trading opportunities based on price and volume action. It provides technical insights to the Decision Model.
*   **MCP Tool Interactions:** The `market_analysis_model` directly interacts with the `FinancialDataMCP` (for retrieving historical and real-time market data), the `TimeSeriesMCP` (for calculating technical indicators, detecting patterns, and identifying support/resistance), and the `RedisMCP` (for caching market data and scan results, and sending feedback/analysis reports to other models via streams). It sends feedback to the SelectionModel and analysis reports to the DecisionModel.

### `risk_assessment_model`

*   **Purpose and Functionality:** The `risk_assessment_model` is responsible for evaluating and managing risk within the trading system. It calculates portfolio and position risk metrics (like VaR, Expected Shortfall, Volatility, Beta), performs risk attribution, generates market scenarios for stress testing, monitors risk limits, and provides risk-based recommendations. It acts as a central hub for risk information, consolidating data from other models.
*   **MCP Tool Interactions:** The `risk_assessment_model` directly interacts with the `RiskAnalysisMCP` (for core risk calculations, scenario generation, attribution, and optimization) and the `RedisMCP` (for state management, storing risk data, monitoring risk limits, and inter-model communication). It receives reports from the SentimentAnalysisModel, FundamentalAnalysisModel, MarketAnalysisModel, and Technical Analysis (implicitly via MarketAnalysisModel) by reading from their respective Redis streams/keys and publishes consolidated risk packages to the DecisionModel via a Redis stream. It also monitors trade events from the TradeModel via a Redis stream to update its internal portfolio state.

### `select_model`

*   **Purpose and Functionality:** The `select_model` is responsible for identifying a universe of potential trading candidates based on predefined criteria such as price, volume, liquidity, and potentially technical signals and unusual activity. It filters and ranks stocks to provide a list of promising candidates for further analysis by other models.
*   **MCP Tool Interactions:** The `select_model` directly interacts with the `TradingMCP` (for account information like buying power), the `FinancialDataMCP` (for retrieving market data, quotes, and unusual activity data), the `TimeSeriesMCP` (for calculating technical indicators and analyzing price data), and the `RedisMCP` (for storing selected candidates and receiving feedback from other models via streams). It receives feedback from the SentimentAnalysisModel, MarketAnalysisModel, and FundamentalAnalysisModel.

### `sentiment_analysis_model`

*   **Purpose and Functionality:** The `sentiment_analysis_model` analyzes text data from various sources (news, social media) to extract relevant financial entities and determine the sentiment associated with them. It provides sentiment scores and insights to other models, particularly the Decision Model and Risk Assessment Model.
*   **MCP Tool Interactions:** The `sentiment_analysis_model` directly interacts with the `FinancialDataMCP` (which now includes functionality for news/social data retrieval, entity extraction, and sentiment scoring) and the `RedisMCP` (for caching sentiment data and publishing sentiment analysis reports to a Redis stream for other models to consume). It also sends feedback to the SelectionModel via a Redis stream.

### `trade_model`

*   **Purpose and Functionality:** The `trade_model` is responsible for executing trading decisions received from the Decision Model and monitoring the status of open positions. It interacts with the trading platform (Alpaca) to submit orders, manage positions, and retrieve account information. It also tracks trade performance and publishes trade events.
*   **MCP Tool Interactions:** The `trade_model` directly interacts with the `TradingMCP` (for all core trading functionality like submitting orders, getting positions, and retrieving account info), the `FinancialDataMCP` (for retrieving market data like latest quotes needed for monitoring), the `TimeSeriesMCP` (for analysis like slippage calculation or peak detection for exit signals), and the `RedisMCP` (for state management like daily capital usage, monitoring open positions, and publishing trade events and account/position updates to streams/keys for other models to consume).

## MCP Tools

### `polygon_news_mcp`

*   **Purpose and Functionality:** This MCP server provides access to financial news data specifically from the Polygon.io News API. Its primary function is to fetch news articles based on various criteria (latest, by ticker, by sector, by keywords) and potentially perform basic sentiment analysis of the news text.
*   **Interactions:** Primarily interacts with the external Polygon.io News API. It is used by models like `sentiment_analysis_model` and potentially others that require news data.

### `polygon_rest_mcp`

*   **Purpose and Functionality:** This MCP server provides access to a wide range of historical and real-time (snapshot) market data from the Polygon.io REST API. This includes historical bars (OHLCV), trades, quotes, company fundamentals, stock splits, dividends, and market status.
*   **Interactions:** Primarily interacts with the external Polygon.io REST API. It is used by models like `financial_data_mcp`, `market_analysis_model`, `select_model`, and `trade_model` to retrieve various types of market data.

### `polygon_ws_mcp`

*   **Purpose and Functionality:** This MCP server provides access to real-time market data streams from the Polygon.io WebSocket API. It allows subscribing to streams for trades, quotes, and aggregate bars (minute/second) to receive low-latency real-time market data.
*   **Interactions:** Primarily interacts with the external Polygon.io WebSocket API. It is used by models that require real-time data feeds, such as the `trade_model` for monitoring or potentially the `market_analysis_model` for real-time indicator calculations.

### `reddit_mcp`

*   **Purpose and Functionality:** This MCP server provides access to data from the Reddit API, focusing on financial subreddits. It can retrieve posts, comments, search for ticker mentions, and perform sentiment analysis on Reddit content.
*   **Interactions:** Primarily interacts with the external Reddit API (via PRAW). It is used by models like `sentiment_analysis_model` to gather social sentiment data.

### `unusual_whales_mcp`

*   **Purpose and Functionality:** This MCP server provides access to specialized options flow, unusual activity, and dark pool data from the Unusual Whales API. This data is often used to identify significant options trading activity or institutional order flow.
*   **Interactions:** Primarily interacts with the external Unusual Whales API. It is used by models like `select_model` and potentially `risk_assessment_model` to identify unusual trading patterns.

### `yahoo_finance_mcp`

*   **Purpose and Functionality:** This MCP server provides an alternative source for historical stock data, company information, financial statements, and basic market data using the Yahoo Finance API (via `yfinance`). It can serve as a backup or supplementary data source.
*   **Interactions:** Primarily interacts with the external Yahoo Finance API (via `yfinance`). It is used by models like `financial_data_mcp` and potentially others that need historical data or fundamental information.

### `yahoo_news_mcp`

*   **Purpose and Functionality:** This MCP server provides an alternative source for financial news articles and basic news sentiment analysis using the Yahoo News API (via `yfinance`). It can serve as a backup or supplementary news source.
*   **Interactions:** Primarily interacts with the external Yahoo Finance API (via `yfinance`). It is used by models like `sentiment_analysis_model` to gather news data.

### `redis_mcp`

*   **Purpose and Functionality:** This MCP server provides a standardized interface for interacting with a Redis server. It supports basic key-value operations, hash operations, list operations, sorted set operations, JSON operations, and Pub/Sub messaging. Its primary purpose is to facilitate inter-model communication, store application state, cache data, and manage message queues/streams.
*   **Interactions:** Interacts directly with a Redis server. It is used by almost all other models (`context_model`, `decision_model`, `fundamental_analysis_model`, `market_analysis_model`, `risk_assessment_model`, `select_model`, `sentiment_analysis_model`, `trade_model`) for various data storage, retrieval, and messaging tasks.

### `document_analysis_mcp`

*   **Purpose and Functionality:** This MCP server is designed for processing and analyzing financial documents (like PDFs). It can extract text, understand document layout, generate embeddings for text chunks, reformulate queries for retrieval, and incorporate relevance feedback.
*   **Interactions:** Internally uses libraries like PyMuPDF (fitz) for PDF processing and potentially Hugging Face Transformers (LayoutLM, BERT-Fin) for layout understanding and embeddings. It is used by models like `context_model` to process and index financial documents.

### `financial_data_mcp`

*   **Purpose and Functionality:** This MCP server acts as an integrated system that combines data retrieval from various sources (Polygon, Yahoo, Unusual Whales) with initial processing capabilities like sentiment analysis (using FinBERT) and potentially predictive modeling (using XGBoost). It provides a unified interface for accessing diverse financial data and some pre-computed analytics.
*   **Interactions:** Internally interacts with `polygon_rest_mcp`, `polygon_news_mcp`, `yahoo_finance_mcp`, `yahoo_news_mcp`, and `unusual_whales_mcp` to fetch raw data. It also uses libraries like `transformers` (FinBERT) and `xgboost` for internal processing. It is used by models like `decision_model`, `fundamental_analysis_model`, `market_analysis_model`, `risk_assessment_model`, and `select_model` to get processed financial data and insights.

### `risk_analysis_mcp`

*   **Purpose and Functionality:** This MCP server provides advanced risk analysis capabilities. It can calculate various risk metrics (VaR, CVaR, volatility, beta), perform risk attribution, optimize portfolio weights based on risk/return objectives, analyze execution slippage, and generate market scenarios for stress testing.
*   **Interactions:** Internally uses libraries like PyPortfolioOpt (`pypfopt`), SciPy (`scipy`), Statsmodels (`statsmodels`), Prophet (`prophet`), and XGBoost (`xgboost`) for its calculations. It is used by models like `decision_model` and `risk_assessment_model` to evaluate and manage risk.

### `time_series_mcp`

*   **Purpose and Functionality:** This MCP server specializes in time series analysis of financial data. It can calculate technical indicators (using TA-Lib), detect chart patterns, identify support and resistance levels, detect statistical drift or regime changes, and generate forecasts using various time series models (ARIMA, Prophet, potentially others).
*   **Interactions:** Internally uses libraries like TA-Lib (`talib`), NumPy (`numpy`), Pandas (`pandas`), SciPy (`scipy`), Statsmodels (`statsmodels`), Prophet (`prophet`), and TensorFlow/Keras (`tensorflow`) for its analysis and forecasting. It is used by models like `market_analysis_model`, `select_model`, and `trade_model` to analyze price and other time-series data.