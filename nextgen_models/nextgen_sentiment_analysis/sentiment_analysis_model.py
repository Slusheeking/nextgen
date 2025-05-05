"""
Sentiment Analysis Model

This module defines the SentimentAnalysisModel, responsible for processing text data
to extract entities and determine sentiment scores using dedicated MCP tools.
It also handles storage and retrieval of sentiment data for the trading system.
"""

import asyncio
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# For Redis integration
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# MCP tools
from mcp_tools.analysis_mcp.sentiment_scoring_mcp import SentimentScoringMCP
from mcp_tools.analysis_mcp.entity_extraction_mcp import EntityExtractionMCP
from mcp_tools.data_mcp.polygon_news_mcp import PolygonNewsMCP
from mcp_tools.data_mcp.reddit_mcp import RedditMCP
from mcp_tools.data_mcp.yahoo_news_mcp import YahooNewsMCP

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function


class SentimentAnalysisModel:
    """
    Analyzes text data (e.g., news, social media) to extract relevant financial entities
    and calculate sentiment scores associated with them.
    This model interacts with EntityExtractionMCP and SentimentScoringMCP.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SentimentAnalysisModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - entity_extraction_config: Config for EntityExtractionMCP.
                - sentiment_scoring_config: Config for SentimentScoringMCP.
                - redis_config: Config for RedisMCP.
                - polygon_news_config: Config for PolygonNewsMCP.
                - reddit_config: Config for RedditMCP.
                - yahoo_news_config: Config for YahooNewsMCP.
                - symbol_entity_mapping: Optional mapping of entities to symbols.
                - batch_size: Size of batches for processing (default: 10).
                - sentiment_ttl: Time-to-live for sentiment data in seconds (default: 86400 - 1 day).
                - llm_config: Configuration for AutoGen LLM.
        """
        init_start_time = time.time()
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-sentiment-analysis-model")

        # Initialize StockChartGenerator
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")

        # Initialize counters for sentiment analysis metrics
        self.texts_analyzed_count = 0
        self.entities_extracted_count = 0
        self.sentiment_scores_generated_count = 0
        self.positive_sentiment_count = 0
        self.negative_sentiment_count = 0
        self.neutral_sentiment_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0 # Errors during analysis process


        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_sentiment_analysis", "sentiment_analysis_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}")
                    self.execution_errors += 1
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}")
                    self.execution_errors += 1
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}")
                self.config = {}
        else:
            self.config = config

        # Initialize MCP clients
        self.entity_extraction_mcp = EntityExtractionMCP(
            self.config.get("entity_extraction_config")
        )
        self.sentiment_scoring_mcp = SentimentScoringMCP(
            self.config.get("sentiment_scoring_config")
        )
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))

        # Initialize news data sources
        self.polygon_news_mcp = PolygonNewsMCP(self.config.get("polygon_news_config"))
        self.reddit_mcp = RedditMCP(self.config.get("reddit_config"))
        self.yahoo_news_mcp = YahooNewsMCP(self.config.get("yahoo_news_config"))

        # For backward compatibility with existing code
        self.entity_client = self.entity_extraction_mcp
        self.sentiment_client = self.sentiment_scoring_mcp
        self.redis_client = self.redis_mcp

        # Configuration parameters
        self.batch_size = self.config.get("batch_size", 10)
        self.sentiment_ttl = self.config.get("sentiment_ttl", 86400)  # Default: 1 day

        # Entity to symbol mapping (can be updated dynamically)
        self.symbol_entity_mapping = self.config.get("symbol_entity_mapping", {})

        # Redis keys for data storage
        self.redis_keys = {
            "sentiment_data": "sentiment:data",  # Overall sentiment data
            "entity_sentiment": "sentiment:entity:",  # Prefix for entity-specific sentiment
            "symbol_sentiment": "sentiment:symbol:",  # Prefix for symbol-specific sentiment
            "sentiment_history": "sentiment:history:",  # Prefix for historical sentiment
            "latest_analysis": "sentiment:latest_analysis",  # Latest analysis timestamp
            "selection_data": "selection:data",  # Selection model data
            "selection_request": "selection:request:",  # Selection request prefix
            "selection_feedback": "sentiment:selection_feedback",  # Feedback to selection model
        }

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        self.logger.info("SentimentAnalysisModel initialized.")
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("sentiment_analysis_model.initialization_time_ms", init_duration)


    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen.

        Returns:
            LLM configuration dictionary
        """
        llm_config = self.config.get("llm_config", {})

        # Default configuration if not provided
        if not llm_config:
            llm_config = {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            }

        return {
            "config_list": llm_config.get("config_list", []),
            "temperature": llm_config.get("temperature", 0.1),
            "timeout": llm_config.get("timeout", 600),
            "seed": 42,  # Adding seed for reproducibility
        }

    def _setup_agents(self) -> Dict[str, Agent]:
        """
        Initialize AutoGen agents for sentiment analysis.

        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the sentiment analysis assistant agent
        agents["sentiment_assistant"] = AssistantAgent(
            name="SentimentAssistantAgent",
            system_message="""You are a financial sentiment analysis specialist. Your role is to:
            1. Analyze news, social media, and other text sources to extract sentiment
            2. Identify relevant entities (companies, products, people, etc.)
            3. Determine sentiment scores for overall text and specific entities
            4. Provide insights on how sentiment might impact trading decisions

            You have tools for extracting entities, scoring sentiment, and retrieving historical sentiment data.
            Always provide clear reasoning for your sentiment assessments.""",
            llm_config=self.llm_config,
            description="A specialist in financial sentiment analysis",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="SentimentToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        agents["user_proxy"] = user_proxy

        return agents

    def _register_functions(self):
        """
        Register functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        sentiment_assistant = self.agents["sentiment_assistant"]

        # Register entity extraction functions
        @register_function(
            name="extract_entities",
            description="Extract entities from text",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def extract_entities(text: str) -> Dict[str, Any]:
            result = await self._call_entity_extraction(text)
            return result or {"entities": [], "error": "Entity extraction failed"}

        # Register sentiment scoring functions
        @register_function(
            name="score_sentiment",
            description="Score sentiment for text and entities",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def score_sentiment(
            text: str, entities: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            result = await self._call_sentiment_scoring(text, entities)
            return result or {
                "overall_sentiment": None,
                "entity_sentiments": {},
                "error": "Sentiment scoring failed",
            }

        # Register full analysis function
        @register_function(
            name="analyze_text",
            description="Perform full sentiment analysis on text",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def analyze_text(text: str) -> Dict[str, Any]:
            result = await self._analyze_single_text(text)
            return result

        # Register batch processing function
        @register_function(
            name="process_news_batch",
            description="Process a batch of news items",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def process_news_batch(
            news_items: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            result = await self.process_news_batch(news_items)
            return result

        # Register symbol sentiment retrieval
        @register_function(
            name="get_symbol_sentiment",
            description="Get aggregated sentiment for a symbol",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def get_symbol_sentiment(
            symbol: str, lookback_hours: int = 24
        ) -> Dict[str, Any]:
            result = await self.get_symbol_sentiment(symbol, lookback_hours)
            return result

        # Register all symbols sentiment retrieval
        @register_function(
            name="get_all_symbols_sentiment",
            description="Get sentiment for all symbols",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def get_all_symbols_sentiment(
            lookback_hours: int = 24,
        ) -> Dict[str, Dict[str, Any]]:
            result = await self.get_all_symbols_sentiment(lookback_hours)
            return result

        # Register sentiment history retrieval
        @register_function(
            name="get_sentiment_history",
            description="Get historical sentiment for a symbol",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def get_sentiment_history(symbol: str, days: int = 7) -> Dict[str, Any]:
            result = await self.get_sentiment_history(symbol, days)
            return result

        # Register symbol-entity mapping update
        @register_function(
            name="update_symbol_entity_mapping",
            description="Update the mapping between entities and symbols",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def update_symbol_entity_mapping(
            mapping: Dict[str, str],
        ) -> Dict[str, Any]:
            result = await self.update_symbol_entity_mapping(mapping)
            return result

        # Register direct news fetching functions
        @register_function(
            name="fetch_latest_news",
            description="Fetch latest news for a symbol or topic",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        async def fetch_latest_news(
            symbol: Optional[str] = None, topic: Optional[str] = None, limit: int = 10
        ) -> Dict[str, Any]:
            result = await self.fetch_latest_news(symbol, topic, limit)
            return result

        # Register selection model integration functions
        @register_function(
            name="get_selection_data",
            description="Get data from the Selection Model",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def get_selection_data() -> Dict[str, Any]:
            return self.get_selection_data()

        @register_function(
            name="send_feedback_to_selection",
            description="Send sentiment feedback to the Selection Model",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def send_feedback_to_selection(sentiment_data: Dict[str, Any]) -> bool:
            return self.send_feedback_to_selection(sentiment_data)

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        sentiment_assistant = self.agents["sentiment_assistant"]

        # Define MCP tool access functions
        @register_function(
            name="use_entity_extraction_tool",
            description="Use a tool provided by the Entity Extraction MCP server",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def use_entity_extraction_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.entity_extraction_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_sentiment_scoring_tool",
            description="Use a tool provided by the Sentiment Scoring MCP server",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def use_sentiment_scoring_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.sentiment_scoring_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_polygon_news_tool",
            description="Use a tool provided by the Polygon News MCP server",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def use_polygon_news_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.polygon_news_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_reddit_tool",
            description="Use a tool provided by the Reddit MCP server",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def use_reddit_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.reddit_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_yahoo_news_tool",
            description="Use a tool provided by the Yahoo News MCP server",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        def use_yahoo_news_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.yahoo_news_mcp.call_tool(tool_name, arguments)

    async def _call_entity_extraction(self, text: str) -> Optional[Dict[str, Any]]:
        """Helper to call the entity extraction service via its client."""
        try:
            # Assuming the client has an async method like 'extract_entities'
            if hasattr(
                self.entity_client, "extract_entities"
            ) and asyncio.iscoroutinefunction(self.entity_client.extract_entities):
                # The actual call to the MCP tool client
                result = await self.entity_client.extract_entities(text=text)
                if result and not result.get("error"):
                    return result
                else:
                    self.logger.error(
                        f"Entity extraction MCP call failed: {result.get('error')}"
                    )
                    return None
            else:
                self.logger.error(
                    "Configured entity client lacks an async 'extract_entities' method."
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Error calling entity extraction service: {e}", exc_info=True
            )
            return None

    async def _call_sentiment_scoring(
        self, text: str, entities: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Helper to call the sentiment scoring service via its client."""
        try:
            # Assuming the client has an async method like 'score_sentiment'
            if hasattr(
                self.sentiment_client, "score_sentiment"
            ) and asyncio.iscoroutinefunction(self.sentiment_client.score_sentiment):
                # The actual call to the MCP tool client
                result = await self.sentiment_client.score_sentiment(
                    text=text, entities=entities
                )
                if result and not result.get("error"):
                    return result
                else:
                    self.logger.error(
                        f"Sentiment scoring MCP call failed: {result.get('error')}"
                    )
                    return None
            else:
                self.logger.error(
                    "Configured sentiment client lacks an async 'score_sentiment' method."
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Error calling sentiment scoring service: {e}", exc_info=True
            )
            return None

    async def analyze_sentiment(
        self, text_data: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes sentiment for given text data using configured MCP clients.

        Args:
            text_data: A single string or a list of strings containing the text to analyze.

        Returns:
            A list of dictionaries, each containing the analysis results for one input text:
            {
                "input_text": str,
                "overall_sentiment": {"label": str, "score": float} | None,
                "entities": [
                    {"text": str, "type": str, "sentiment": {"label": str, "score": float}}
                ],
                "error": Optional[str]
            }
            Returns None for sentiment fields if the respective MCP call fails.
        """
        if isinstance(text_data, str):
            texts = [text_data]
        elif isinstance(text_data, list):
            texts = text_data
        else:
            self.logger.error(f"Invalid input type for text_data: {type(text_data)}")
            raise TypeError("text_data must be a string or a list of strings.")

        #
        # Use asyncio.gather to run analyses concurrently if multiple texts are
        # provided
        tasks = [self._analyze_single_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling potential exceptions from gather
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.error(
                    f"Exception during sentiment analysis for text index {i}: {res}",
                    exc_info=res,
                )
                final_results.append(
                    {
                        "input_text": texts[i],
                        "overall_sentiment": None,
                        "entities": [],
                        "error": f"An unexpected error occurred during analysis: {str(res)}",
                    }
                )
            else:
                final_results.append(res)

        return final_results

    async def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """Analyzes a single piece of text."""
        analysis_result = {
            "input_text": text,
            "overall_sentiment": None,
            "entities": [],
            "error": None,
        }
        entity_error = None
        sentiment_error = None

        try:
            # 1. Extract Entities
            entity_result = await self._call_entity_extraction(text)
            extracted_entities = []
            entity_texts = []
            entity_map = {}
            if entity_result and entity_result.get("entities"):
                extracted_entities = entity_result["entities"]
                entity_texts = [
                    entity["text"]
                    for entity in extracted_entities
                    if entity.get("text")
                ]
                entity_map = {
                    entity["text"]: entity
                    for entity in extracted_entities
                    if entity.get("text")
                }
            elif entity_result is None or entity_result.get("error"):
                entity_error = f"Entity extraction failed: {entity_result.get('error', 'Unknown error') if entity_result else 'Service call failed'}"
                self.logger.warning(f"{entity_error} for text: '{text[:50]}...'")

            # 2. Score Sentiment (Overall and for Entities if any)
            sentiment_result = await self._call_sentiment_scoring(
                text, entities=entity_texts if entity_texts else None
            )

            if sentiment_result:
                analysis_result["overall_sentiment"] = sentiment_result.get(
                    "overall_sentiment"
                )

                # Merge entity sentiment with extracted entity details
                entity_sentiments = sentiment_result.get("entity_sentiments", {})
                processed_entities = []
                # Use the entity_map created earlier
                for entity_text, sentiment_score in entity_sentiments.items():
                    entity_detail = entity_map.get(
                        entity_text, {"text": entity_text, "type": "UNKNOWN"}
                    )  # Fallback
                    processed_entities.append(
                        {
                            "text": entity_text,
                            "type": entity_detail.get("type"),
                            "sentiment": sentiment_score,
                        }
                    )
                analysis_result["entities"] = processed_entities
            else:
                sentiment_error = f"Sentiment scoring failed: {sentiment_result.get('error', 'Unknown error') if sentiment_result else 'Service call failed'}"
                self.logger.warning(f"{sentiment_error} for text: '{text[:50]}...'")

            # Consolidate errors if any occurred
            errors = [err for err in [entity_error, sentiment_error] if err]
            if errors:
                analysis_result["error"] = "; ".join(errors)

        except Exception as e:
            self.logger.error(
                f"Unexpected error during single text analysis: '{text[:100]}...': {e}",
                exc_info=True,
            )
            analysis_result["error"] = (
                f"An unexpected internal error occurred: {str(e)}"
            )

        return analysis_result

    async def process_news_batch(
        self, news_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a batch of news items and analyze sentiment.

        Args:
            news_items: List of news items, each containing at least:
                - text: The news text
                - source: Source of the news
                - published_at: Publication timestamp
                - url: Optional URL to the news

        Returns:
            Dictionary with processing results and statistics
        """
        self.logger.info(f"Processing batch of {len(news_items)} news items")

        # Initialize tracking variables
        start_time = time.time()
        processed_items = []

        self.logger.info(
            f"Processing batch of {len(news_items)} news items",
            extra={
                "component": "sentiment_analysis",
                "action": "process_news_batch",
                "batch_size": len(news_items),
            },
        )

        # Process each news item
        for news_item in news_items:
            try:
                # Extract text from the news item
                text = news_item.get("text", "")
                if not text:
                    self.logger.warning(
                        f"Skipping news item without text: {news_item.get('title', 'Unknown')}"
                    )
                    continue

                # Analyze sentiment
                result = await self._analyze_single_text(text)

                # Add sentiment analysis to the news item
                news_item["sentiment_analysis"] = result

                # Extract symbols from entities
                symbols = self._extract_symbols_from_entities(
                    result.get("entities", [])
                )
                if symbols:
                    news_item["related_symbols"] = list(symbols)

                # Store in Redis if available
                if self.redis_client:
                    await self._store_sentiment_result(result, news_item)

                # Add to processed items
                processed_items.append(news_item)

                # (Metric update removed: dynamic metrics are not supported in the new MonitoringManager)

            except Exception as e:
                self.logger.error(f"Error processing news item: {e}", exc_info=True)

        # Calculate processing time
        processing_time = time.time() - start_time

        # (Processing time metric update removed: dynamic metrics are not supported in the new MonitoringManager)

        # Update latest analysis timestamp
        if self.redis_client:
            self.redis_client.set_json(
                self.redis_keys["latest_analysis"],
                {
                    "timestamp": datetime.now().isoformat(),
                    "count": len(processed_items),
                },
            )

        return {
            "processed": len(processed_items),
            "processing_time": processing_time,
            "items": processed_items,
            "symbols_found": self._count_unique_symbols(processed_items),
        }

    def _extract_symbols_from_entities(
        self, entities: List[Dict[str, Any]]
    ) -> Set[str]:
        """Extract trading symbols from entity list using mapping."""
        symbols = set()

        for entity in entities:
            entity_text = entity.get("text", "").upper()
            entity_type = entity.get("type", "")

            # Direct match if entity is a stock ticker
            if entity_type == "TICKER" or entity_type == "STOCK_TICKER":
                symbols.add(entity_text)

            # Look up in mapping
            elif entity_text in self.symbol_entity_mapping:
                symbols.add(self.symbol_entity_mapping[entity_text])

        return symbols

    def _count_unique_symbols(
        self, processed_items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count occurrences of each symbol in processed items."""
        symbol_counts = {}

        for item in processed_items:
            for symbol in item.get("related_symbols", []):
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return symbol_counts

    async def _store_sentiment_result(
        self, sentiment_result: Dict[str, Any], news_item: Dict[str, Any]
    ) -> None:
        """Store sentiment result in Redis."""
        if not self.redis_client:
            return

        try:
            # Generate a unique ID for this analysis
            analysis_id = (
                f"sentiment:{int(time.time())}:{hash(news_item.get('text', '')[:100])}"
            )

            # Store the full analysis
            self.redis_client.set_json(
                f"sentiment:analysis:{analysis_id}",
                {
                    "sentiment": sentiment_result,
                    "news_item": news_item,
                    "timestamp": datetime.now().isoformat(),
                },
                ex=self.sentiment_ttl,
            )

            # Update entity-specific sentiment
            for entity in sentiment_result.get("entities", []):
                entity_text = entity.get("text")
                if entity_text:
                    entity_key = f"{self.redis_keys['entity_sentiment']}{entity_text}"
                    self.redis_client.add_to_sorted_set(
                        entity_key, analysis_id, int(time.time())
                    )

            # Update symbol-specific sentiment
            for symbol in news_item.get("related_symbols", []):
                symbol_key = f"{self.redis_keys['symbol_sentiment']}{symbol}"
                self.redis_client.add_to_sorted_set(
                    symbol_key, analysis_id, int(time.time())
                )

        except Exception as e:
            self.logger.error(
                f"Error storing sentiment result in Redis: {e}", exc_info=True
            )

    async def get_symbol_sentiment(
        self, symbol: str, lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a specific symbol.

        Args:
            symbol: Trading symbol to get sentiment for
            lookback_hours: Hours to look back for sentiment data

        Returns:
            Aggregated sentiment data for the symbol
        """
        if not self.redis_client:
            return {"error": "Redis client not available"}

        try:
            symbol_key = f"{self.redis_keys['symbol_sentiment']}{symbol}"

            # Get analysis IDs for this symbol from the last N hours
            min_time = int(time.time() - (lookback_hours * 3600))
            analysis_ids = self.redis_client.get_sorted_set_members_by_score(
                symbol_key, min_score=min_time
            )

            if not analysis_ids:
                return {
                    "symbol": symbol,
                    "sentiment": None,
                    "entity_count": 0,
                    "news_count": 0,
                    "lookback_hours": lookback_hours,
                }

            # Fetch all analyses
            analyses = []
            for analysis_id in analysis_ids:
                analysis = self.redis_client.get_json(
                    f"sentiment:analysis:{analysis_id}"
                )
                if analysis:
                    analyses.append(analysis)

            # Aggregate sentiment
            return self._aggregate_symbol_sentiment(symbol, analyses, lookback_hours)

        except Exception as e:
            self.logger.error(
                f"Error getting sentiment for symbol {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    def _aggregate_symbol_sentiment(
        self, symbol: str, analyses: List[Dict[str, Any]], lookback_hours: int
    ) -> Dict[str, Any]:
        """Aggregate sentiment from multiple analyses for a symbol."""
        if not analyses:
            return {
                "symbol": symbol,
                "sentiment": None,
                "entity_count": 0,
                "news_count": 0,
                "lookback_hours": lookback_hours,
            }

        # Extract sentiment scores
        overall_scores = []
        entity_scores = []
        sources = set()

        for analysis in analyses:
            sentiment_data = analysis.get("sentiment", {})
            news_item = analysis.get("news_item", {})

            # Overall sentiment
            overall = sentiment_data.get("overall_sentiment", {})
            if overall and "score" in overall:
                overall_scores.append(float(overall["score"]))

            # Entity sentiment
            for entity in sentiment_data.get("entities", []):
                if (
                    entity.get("text", "").upper() == symbol
                    or entity.get("text") in self.symbol_entity_mapping
                ):
                    sentiment = entity.get("sentiment", {})
                    if sentiment and "score" in sentiment:
                        entity_scores.append(float(sentiment["score"]))

            # Track sources
            if "source" in news_item:
                sources.add(news_item["source"])

        # Calculate aggregated sentiment
        avg_overall = (
            sum(overall_scores) / len(overall_scores) if overall_scores else None
        )
        avg_entity = sum(entity_scores) / len(entity_scores) if entity_scores else None

        # Determine sentiment label
        sentiment_label = "NEUTRAL"
        sentiment_score = avg_entity if avg_entity is not None else avg_overall

        if sentiment_score is not None:
            if sentiment_score > 0.2:
                sentiment_label = "POSITIVE"
            elif sentiment_score < -0.2:
                sentiment_label = "NEGATIVE"

        return {
            "symbol": symbol,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment_score,
                "overall_score": avg_overall,
                "entity_score": avg_entity,
            },
            "entity_count": len(entity_scores),
            "news_count": len(analyses),
            "sources": list(sources),
            "lookback_hours": lookback_hours,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_all_symbols_sentiment(
        self, lookback_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sentiment for all symbols that have recent data.

        Args:
            lookback_hours: Hours to look back for sentiment data

        Returns:
            Dictionary mapping symbols to their sentiment data
        """
        if not self.redis_client:
            return {"error": "Redis client not available"}

        try:
            # Get all symbol keys
            symbol_keys = self.redis_client.keys(
                f"{self.redis_keys['symbol_sentiment']}*"
            )

            if not symbol_keys:
                return {}

            # Extract symbols from keys
            symbols = [
                key.replace(self.redis_keys["symbol_sentiment"], "")
                for key in symbol_keys
            ]

            # Get sentiment for each symbol
            results = {}
            for symbol in symbols:
                sentiment = await self.get_symbol_sentiment(symbol, lookback_hours)
                if sentiment and not sentiment.get("error"):
                    results[symbol] = sentiment

            return {
                "symbols": results,
                "count": len(results),
                "lookback_hours": lookback_hours,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(
                f"Error getting sentiment for all symbols: {e}", exc_info=True
            )
            return {"error": str(e)}

    async def update_symbol_entity_mapping(
        self, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Update the mapping between entities and symbols.

        Args:
            mapping: Dictionary mapping entity names to symbols

        Returns:
            Status of the update operation
        """
        try:
            # Update the mapping
            self.symbol_entity_mapping.update(mapping)

            # Store in Redis if available
            if self.redis_client:
                self.redis_client.set_json(
                    "sentiment:symbol_entity_mapping", self.symbol_entity_mapping
                )

            return {
                "status": "success",
                "updated_count": len(mapping),
                "total_mappings": len(self.symbol_entity_mapping),
            }

        except Exception as e:
            self.logger.error(
                f"Error updating symbol-entity mapping: {e}", exc_info=True
            )
            return {"status": "error", "error": str(e)}

    async def get_sentiment_history(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical sentiment data for a symbol.

        Args:
            symbol: Trading symbol to get history for
            days: Number of days to look back

        Returns:
            Historical sentiment data by day
        """
        if not self.redis_client:
            return {"error": "Redis client not available"}

        try:
            # Calculate daily sentiment for each of the past N days
            history = []
            now = datetime.now()

            for day in range(days):
                day_start = now - timedelta(days=day + 1)
                day_end = now - timedelta(days=day)

                # Get sentiment for this day
                lookback_hours = 24
                day_sentiment = await self._get_sentiment_for_timerange(
                    symbol, int(day_start.timestamp()), int(day_end.timestamp())
                )

                if day_sentiment:
                    history.append(
                        {
                            "date": day_start.strftime("%Y-%m-%d"),
                            "sentiment": day_sentiment,
                        }
                    )

            return {"symbol": symbol, "history": history, "days": days}

        except Exception as e:
            self.logger.error(
                f"Error getting sentiment history for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    async def _get_sentiment_for_timerange(
        self, symbol: str, start_time: int, end_time: int
    ) -> Optional[Dict[str, Any]]:
        """Get sentiment for a specific time range."""
        symbol_key = f"{self.redis_keys['symbol_sentiment']}{symbol}"

        # Get analysis IDs for this symbol in the time range
        analysis_ids = self.redis_client.get_sorted_set_members_by_score(
            symbol_key, min_score=start_time, max_score=end_time
        )

        if not analysis_ids:
            return None

        # Fetch all analyses
        analyses = []
        for analysis_id in analysis_ids:
            analysis = self.redis_client.get_json(f"sentiment:analysis:{analysis_id}")
            if analysis:
                analyses.append(analysis)

        # Calculate hours in the time range
        hours = (end_time - start_time) / 3600

        # Aggregate sentiment
        return self._aggregate_symbol_sentiment(symbol, analyses, hours)

    async def fetch_latest_news(
        self, symbol: Optional[str] = None, topic: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Fetch latest news for a symbol or topic from various sources.

        Args:
            symbol: Optional stock symbol to fetch news for
            topic: Optional topic to fetch news for
            limit: Maximum number of news items to fetch

        Returns:
            Dictionary with news items from various sources
        """
        results = {"news_items": [], "sources": [], "count": 0}

        try:
            # Fetch from Polygon News if available
            if hasattr(self.polygon_news_mcp, "get_ticker_news"):
                if symbol:
                    polygon_news = self.polygon_news_mcp.get_ticker_news(
                        ticker=symbol, limit=limit
                    )
                    if polygon_news and not polygon_news.get("error"):
                        results["news_items"].extend(polygon_news.get("results", []))
                        results["sources"].append("polygon")
                elif topic:
                    polygon_news = self.polygon_news_mcp.search_news(
                        query=topic, limit=limit
                    )
                    if polygon_news and not polygon_news.get("error"):
                        results["news_items"].extend(polygon_news.get("results", []))
                        results["sources"].append("polygon")

            # Fetch from Yahoo News if available
            if hasattr(self.yahoo_news_mcp, "get_company_news"):
                if symbol:
                    yahoo_news = self.yahoo_news_mcp.get_company_news(
                        symbol=symbol, limit=limit
                    )
                    if yahoo_news and not yahoo_news.get("error"):
                        results["news_items"].extend(yahoo_news.get("items", []))
                        results["sources"].append("yahoo")
                elif topic:
                    yahoo_news = self.yahoo_news_mcp.search_news(
                        query=topic, limit=limit
                    )
                    if yahoo_news and not yahoo_news.get("error"):
                        results["news_items"].extend(yahoo_news.get("items", []))
                        results["sources"].append("yahoo")

            # Fetch from Reddit if available
            if hasattr(self.reddit_mcp, "search_posts"):
                search_term = symbol if symbol else topic
                if search_term:
                    reddit_posts = self.reddit_mcp.search_posts(
                        query=search_term, limit=limit
                    )
                    if reddit_posts and not reddit_posts.get("error"):
                        results["news_items"].extend(reddit_posts.get("posts", []))
                        results["sources"].append("reddit")

            # Process the fetched news
            if results["news_items"]:
                # Sort by publication date (newest first)
                results["news_items"].sort(
                    key=lambda x: x.get("published_at", x.get("created_at", "")),
                    reverse=True,
                )

                # Limit to requested number
                results["news_items"] = results["news_items"][:limit]

                # Update count
                results["count"] = len(results["news_items"])

                # Process sentiment for the news items
                await self.process_news_batch(results["news_items"])

            return results

        except Exception as e:
            self.logger.error(f"Error fetching news: {e}", exc_info=True)
            return {"error": str(e), "news_items": [], "count": 0}

    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get data from the Selection Model.

        Returns:
            Selection Model data
        """
        try:
            return self.redis_mcp.get_json(self.redis_keys["selection_data"]) or {}
        except Exception as e:
            self.logger.error(f"Error getting selection data: {e}")
            return {}

    def send_feedback_to_selection(self, sentiment_data: Dict[str, Any]) -> bool:
        """
        Send sentiment feedback to the Selection Model.

        Args:
            sentiment_data: Sentiment data to send

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp
            feedback = {
                "source": "sentiment_analysis",
                "timestamp": datetime.now().isoformat(),
                "data": sentiment_data,
            }

            # Store in Redis for the Selection Model to pick up
            self.redis_mcp.set_json(self.redis_keys["selection_feedback"], feedback)

            # Add to feedback stream
            self.redis_mcp.add_to_stream("sentiment:feedback", feedback)

            return True
        except Exception as e:
            self.logger.error(f"Error sending feedback to selection model: {e}")
            return False

    def run_sentiment_analysis(self, query: str) -> Dict[str, Any]:
        """
        Run sentiment analysis using AutoGen agents.

        Args:
            query: Query or instruction for sentiment analysis

        Returns:
            Results of the sentiment analysis
        """
        self.logger.info(f"Running sentiment analysis with query: {query}")

        sentiment_assistant = self.agents.get("sentiment_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not sentiment_assistant or not user_proxy:
            return {"error": "AutoGen agents not initialized"}

        try:
            # Initiate chat with the sentiment assistant
            user_proxy.initiate_chat(sentiment_assistant, message=query)

            # Get the last message from the assistant
            last_message = user_proxy.last_message(sentiment_assistant)
            content = last_message.get("content", "")

            # Extract structured data if possible
            try:
                # Find JSON blocks in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    result_str = content[json_start:json_end]
                    result = json.loads(result_str)
                    return result
            except json.JSONDecodeError:
                # Return the raw content if JSON parsing fails
                pass

            return {"analysis": content}

        except Exception as e:
            self.logger.error(f"Error during AutoGen chat: {e}")
            return {"error": str(e)}


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
