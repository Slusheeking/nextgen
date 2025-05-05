"""
Context Model

This module defines the ContextModel, responsible for gathering, processing, and managing
contextual information from various sources to support other NextGen models.
It integrates with data retrieval, document processing, query reformulation, and
relevance feedback MCP tools.
"""

from monitoring.netdata_logger import NetdataLogger
import json
import time
import os
import hashlib
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import Dict, List, Any, Optional

# No longer using MonitoringManager; replaced by NetdataLogger

# For Redis integration
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from monitoring.system_metrics import SystemMetricsCollector

# MCP tools
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP
from mcp_tools.analysis_mcp.document_processing_mcp import DocumentProcessingMCP
from mcp_tools.analysis_mcp.embeddings_mcp import EmbeddingsMCP
from mcp_tools.analysis_mcp.vector_db_mcp import VectorDBMCP
from mcp_tools.analysis_mcp.query_reformulation_mcp import QueryReformulationMCP
from mcp_tools.analysis_mcp.relevance_feedback_mcp import RelevanceFeedbackMCP

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
)


class ContextModel:
    """
    Gathers, processes, and manages contextual information for other NextGen models.

    This model integrates with various MCP tools to retrieve data, process documents,
    reformulate queries, and incorporate relevance feedback to build a comprehensive
    understanding of the market and relevant information.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ContextModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - data_sources_config: Config for various data source MCPs
                - document_processing_config: Config for DocumentProcessingMCP
                - embeddings_config: Config for EmbeddingsMCP
                - vector_db_config: Config for VectorDBMCP
                - query_reformulation_config: Config for QueryReformulationMCP
                - relevance_feedback_config: Config for RelevanceFeedbackMCP
                - redis_config: Config for RedisMCP
                - context_data_ttl: Time-to-live for context data in seconds (default: 3600 - 1 hour)
                - llm_config: Configuration for AutoGen LLM
        """
        init_start_time = time.time()
        # Set up NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-context-model")

        # Initialize system metrics collector
        self.metrics_collector = SystemMetricsCollector(self.logger)

        # Start collecting system metrics
        self.metrics_collector.start()

        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_context_model", "context_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}")
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}")
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}")
                self.config = {}
        else:
            self.config = config

        # Initialize MCP clients
        # Data Sources (example: Polygon News, Reddit, Yahoo Finance/News - add others as needed)
        self.data_sources: Dict[str, BaseDataMCP] = {}
        data_sources_config = self.config.get("data_sources_config", {})
        for source_name, source_config in data_sources_config.items():
            try:
                # Dynamically import and initialize data source MCPs
                module_path = source_config.get("module_path")
                class_name = source_config.get("class_name")
                if module_path and class_name:
                    module = __import__(module_path, fromlist=[class_name])
                    data_source_class = getattr(module, class_name)
                    self.data_sources[source_name] = data_source_class(
                        source_config.get("config")
                    )
                    self.logger.info(f"Initialized data source MCP: {source_name}")
                else:
                    self.logger.warning(
                        f"Skipping data source '{source_name}': module_path or class_name missing in config."
                    )
            except Exception as e:
                self.logger.error(
                    f"Error initializing data source MCP '{source_name}': {e}"
                )

        self.document_processing_mcp = DocumentProcessingMCP(
            self.config.get("document_processing_config")
        )
        self.embeddings_mcp = EmbeddingsMCP(self.config.get("embeddings_config"))
        self.vector_db_mcp = VectorDBMCP(self.config.get("vector_db_config"))
        self.query_reformulation_mcp = QueryReformulationMCP(
            self.config.get("query_reformulation_config")
        )
        self.relevance_feedback_mcp = RelevanceFeedbackMCP(
            self.config.get("relevance_feedback_config")
        )
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))

        # Configuration parameters
        self.context_data_ttl = self.config.get(
            "context_data_ttl", 3600
        )  # Default: 1 hour

        # Redis keys for data storage
        self.redis_keys = {
            "context_data": "context:data:",  # Prefix for general contextual data
            "document_chunks": "context:chunks:",  # Prefix for document chunks
            "document_metadata": "context:metadata:",  # Prefix for document metadata
            "query_history": "context:queries:",  # Prefix for query history
            "feedback_data": "context:feedback:",  # Prefix for feedback data
            "latest_context_update": "context:latest_update",  # Latest context update timestamp
        }

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        # Configuration parameters
        self.context_data_ttl = self.config.get(
            "context_data_ttl", 3600
        )  # Default: 1 hour

        # Redis keys for data storage
        self.redis_keys = {
            "context_data": "context:data:",  # Prefix for general contextual data
            "document_chunks": "context:chunks:",  # Prefix for document chunks
            "document_metadata": "context:metadata:",  # Prefix for document metadata
            "query_history": "context:queries:",  # Prefix for query history
            "feedback_data": "context:feedback:",  # Prefix for feedback data
            "latest_context_update": "context:latest_update",  # Latest context update timestamp
        }

        # Initialize MCP clients
        # Data Sources (example: Polygon News, Reddit, Yahoo Finance/News - add others as needed)
        self.data_sources: Dict[str, BaseDataMCP] = {}
        data_sources_config = self.config.get("data_sources_config", {})
        for source_name, source_config in data_sources_config.items():
            try:
                # Dynamically import and initialize data source MCPs
                module_path = source_config.get("module_path")
                class_name = source_config.get("class_name")
                if module_path and class_name:
                    module = __import__(module_path, fromlist=[class_name])
                    data_source_class = getattr(module, class_name)
                    self.data_sources[source_name] = data_source_class(
                        source_config.get("config")
                    )
                    self.logger.info(f"Initialized data source MCP: {source_name}")
                else:
                    self.logger.warning(
                        f"Skipping data source '{source_name}': module_path or class_name missing in config."
                    )
            except Exception as e:
                self.logger.error(
                    f"Error initializing data source MCP '{source_name}': {e}"
                )

        self.document_processing_mcp = DocumentProcessingMCP(
            self.config.get("document_processing_config")
        )
        self.embeddings_mcp = EmbeddingsMCP(self.config.get("embeddings_config"))
        self.vector_db_mcp = VectorDBMCP(self.config.get("vector_db_config"))
        self.query_reformulation_mcp = QueryReformulationMCP(
            self.config.get("query_reformulation_config")
        )
        self.relevance_feedback_mcp = RelevanceFeedbackMCP(
            self.config.get("relevance_feedback_config")
        )
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        self.logger.info("ContextModel initialized.")
        self.logger.gauge("context_model.data_sources_count", len(self.data_sources))
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("context_model.initialization_time_ms", init_duration)


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
        Initialize AutoGen agents for context management.

        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the context assistant agent
        agents["context_assistant"] = AssistantAgent(
            name="ContextAssistantAgent",
            system_message="""You are a financial market context specialist. Your role is to:
            1. Gather relevant information from various data sources (news, social media, reports, etc.)
            2. Process and structure raw information for easy retrieval and analysis
            3. Manage document chunks, embeddings, and metadata in a vector database
            4. Reformulate user queries to improve search results
            5. Incorporate user feedback to refine context and retrieval

            You have tools for data retrieval, document processing, embedding generation,
            vector database operations, query reformulation, and relevance feedback.
            Always aim to provide the most relevant and accurate context.""",
            llm_config=self.llm_config,
            description="A specialist in gathering and managing financial market context",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="ContextToolUser",
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
        context_assistant = self.agents["context_assistant"]

        # Register data retrieval functions (example for Polygon News)
        @register_function(
            name="fetch_data",
            description="Fetch data from a specified data source",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def fetch_data(source_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
            return await self.fetch_data(source_name, query)

        # Register document processing functions
        @register_function(
            name="process_document",
            description="Process a document for embedding generation (clean, extract metadata, and chunk)",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def process_document(
            document: str,
            document_id: Optional[str] = None,
            document_type: Optional[str] = None,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            chunk_strategy: str = "paragraph",
        ) -> Dict[str, Any]:
            result = self.document_processing_mcp.process_document(
                document=document,
                document_id=document_id,
                document_type=document_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_strategy=chunk_strategy,
            )
            return result or {"error": "Document processing failed"}

        @register_function(
            name="deduplicate_chunks",
            description="Remove duplicate or near-duplicate chunks",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def deduplicate_chunks(
            chunks: List[str],
            similarity_threshold: float = 0.9,
            method: str = "jaccard",
        ) -> Dict[str, Any]:
            result = self.document_processing_mcp.deduplicate_chunks(
                chunks=chunks, similarity_threshold=similarity_threshold, method=method
            )
            return result or {"error": "Chunk deduplication failed"}

        # Register embeddings functions
        @register_function(
            name="generate_embeddings",
            description="Generate embeddings for text chunks",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def generate_embeddings(
            chunks: List[str], model: Optional[str] = None
        ) -> Dict[str, Any]:
            result = self.embeddings_mcp.generate_embeddings(chunks=chunks, model=model)
            return result or {"error": "Embedding generation failed"}

        # Register vector database functions
        @register_function(
            name="add_documents_to_vector_db",
            description="Add documents (chunks and embeddings) to the vector database",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def add_documents_to_vector_db(
            documents: List[Dict[str, Any]],
            collection_name: str,
            metadata: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            result = self.vector_db_mcp.add_documents(
                documents=documents, collection_name=collection_name, metadata=metadata
            )
            return result or {"error": "Adding documents to vector DB failed"}

        @register_function(
            name="search_vector_db",
            description="Search the vector database using a query embedding",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def search_vector_db(
            query_embedding: List[float],
            collection_name: str,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            result = self.vector_db_mcp.search(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=top_k,
                filters=filters,
            )
            return result or {"error": "Vector DB search failed"}

        # Register query reformulation functions
        @register_function(
            name="expand_query",
            description="Expand a query with related terms to improve retrieval",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def expand_query(
            query: str,
            domain: Optional[str] = None,
            strategies: Optional[List[str]] = None,
            max_terms: Optional[int] = None,
        ) -> Dict[str, Any]:
            result = self.query_reformulation_mcp.expand_query(
                query=query, domain=domain, strategies=strategies, max_terms=max_terms
            )
            return result or {"error": "Query expansion failed"}

        @register_function(
            name="decompose_query",
            description="Break down a complex query into simpler sub-queries",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def decompose_query(
            query: str, max_sub_queries: Optional[int] = None
        ) -> Dict[str, Any]:
            result = self.query_reformulation_mcp.decompose_query(
                query=query, max_sub_queries=max_sub_queries
            )
            return result or {"error": "Query decomposition failed"}

        # Register relevance feedback functions
        @register_function(
            name="record_explicit_feedback",
            description="Record explicit user feedback on document relevance",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def record_explicit_feedback(
            query_id: str,
            document_id: str,
            rating: float,
            query_text: Optional[str] = None,
            document_text: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            result = self.relevance_feedback_mcp.record_explicit_feedback(
                query_id=query_id,
                document_id=document_id,
                rating=rating,
                query_text=query_text,
                document_text=document_text,
                user_id=user_id,
            )
            return result or {"error": "Recording explicit feedback failed"}

        @register_function(
            name="record_implicit_feedback",
            description="Record implicit user feedback based on interactions",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def record_implicit_feedback(
            query_id: str,
            document_id: str,
            interaction_type: str,
            interaction_value: float,
            query_text: Optional[str] = None,
            document_text: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            result = self.relevance_feedback_mcp.record_implicit_feedback(
                query_id=query_id,
                document_id=document_id,
                interaction_type=interaction_type,
                interaction_value=interaction_value,
                query_text=query_text,
                document_text=document_text,
                user_id=user_id,
            )
            return result or {"error": "Recording implicit feedback failed"}

        @register_function(
            name="rerank_results",
            description="Rerank search results based on feedback",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def rerank_results(
            query_id: str,
            results: List[Dict[str, Any]],
            use_explicit_feedback: bool = True,
            use_implicit_feedback: bool = True,
            feedback_weight: Optional[float] = None,
        ) -> Dict[str, Any]:
            result = self.relevance_feedback_mcp.rerank_results(
                query_id=query_id,
                results=results,
                use_explicit_feedback=use_explicit_feedback,
                use_implicit_feedback=use_implicit_feedback,
                feedback_weight=feedback_weight,
            )
            return result or {"error": "Reranking results failed"}

        # Register data storage and retrieval functions
        @register_function(
            name="store_context_data",
            description="Store general contextual data in Redis",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def store_context_data(key: str, data: Any) -> Dict[str, Any]:
            return await self.store_context_data(key, data)

        @register_function(
            name="get_context_data",
            description="Retrieve general contextual data from Redis",
            caller=context_assistant,
            executor=user_proxy,
        )
        async def get_context_data(key: str) -> Dict[str, Any]:
            return await self.get_context_data(key)

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        context_assistant = self.agents["context_assistant"]

        # Define MCP tool access functions for data sources
        for source_name, data_source_mcp in self.data_sources.items():

            @register_function(
                name=f"use_{source_name}_tool",
                description=f"Use a tool provided by the {source_name} MCP server",
                caller=context_assistant,
                executor=user_proxy,
            )
            def use_data_source_tool(
                tool_name: str, arguments: Dict[str, Any], mcp_client=data_source_mcp
            ) -> Any:
                return mcp_client.call_tool(tool_name, arguments)

        # Define MCP tool access functions for analysis tools
        @register_function(
            name="use_document_processing_tool",
            description="Use a tool provided by the Document Processing MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )
        def use_document_processing_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.document_processing_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_embeddings_tool",
            description="Use a tool provided by the Embeddings MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )
        def use_embeddings_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.embeddings_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_vector_db_tool",
            description="Use a tool provided by the Vector DB MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )
        def use_vector_db_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.vector_db_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_query_reformulation_tool",
            description="Use a tool provided by the Query Reformulation MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )
        def use_query_reformulation_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.query_reformulation_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_relevance_feedback_tool",
            description="Use a tool provided by the Relevance Feedback MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )
        def use_relevance_feedback_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.relevance_feedback_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

    async def fetch_data(
        self, source_name: str, query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fetch data from a specified data source MCP.

        Args:
            source_name: Name of the data source (e.g., 'polygon_news', 'reddit')
            query: Dictionary containing the query parameters for the data source tool

        Returns:
            Dictionary with fetched data or error
        """
        try:
            if source_name not in self.data_sources:
                return {
                    "error": f"Data source '{source_name}' not initialized or supported."
                }

            data_source_mcp = self.data_sources[source_name]

            # Assuming data source MCPs have a generic 'fetch' tool or similar
            # In a real implementation, you might map source_name and query to specific tools
            if hasattr(data_source_mcp, "fetch"):
                result = data_source_mcp.fetch(
                    **query
                )  # Assuming query maps directly to args
                return result
            elif hasattr(data_source_mcp, "call_tool"):
                # Attempt to call a tool specified in the query, e.g., query = {"tool_name": "get_latest_articles", "args": {"symbol": "AAPL"}}
                tool_name = query.get("tool_name")
                tool_args = query.get("args", {})
                if tool_name:
                    result = data_source_mcp.call_tool(tool_name, tool_args)
                    return result
                else:
                    return {
                        "error": f"No tool_name specified for data source '{source_name}'."
                    }
            else:
                return {
                    "error": f"Data source '{source_name}' does not have a 'fetch' or 'call_tool' method."
                }

        except Exception as e:
            self.logger.error(f"Error fetching data from {source_name}: {e}")
            return {"error": str(e)}

    async def process_and_store_document(
        self,
        document: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        collection_name: str = "default_collection",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_strategy: str = "paragraph",
    ) -> Dict[str, Any]:
        """
        Process a document, generate embeddings, and store in the vector database.

        Args:
            document: Document text to process
            document_id: Optional unique identifier for the document
            document_type: Optional type of document
            collection_name: Name of the vector database collection to store in
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            chunk_strategy: Strategy for chunking

        Returns:
            Dictionary with processing and storage results
        """
        self.logger.info(
            f"Processing and storing document (ID: {document_id or 'auto-generated'})"
        )
        start_time = time.time()

        try:
            # 1. Process the document (clean, metadata, chunk)
            process_result = self.document_processing_mcp.process_document(
                document=document,
                document_id=document_id,
                document_type=document_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_strategy=chunk_strategy,
            )

            if "error" in process_result:
                return {
                    "error": f"Document processing failed: {process_result['error']}"
                }

            chunks = process_result.get("chunks", [])
            chunk_metadata = process_result.get("chunk_metadata", [])
            document_metadata = process_result.get("document_metadata", {})
            processed_document_id = process_result.get("document_id")

            if not chunks:
                return {
                    "status": "skipped",
                    "message": "No chunks generated from document.",
                }

            # 2. Generate embeddings for chunks
            embedding_result = self.embeddings_mcp.generate_embeddings(chunks=chunks)

            if "error" in embedding_result:
                return {
                    "error": f"Embedding generation failed: {embedding_result['error']}"
                }

            embeddings = embedding_result.get("embeddings", [])

            if len(chunks) != len(embeddings):
                return {
                    "error": "Mismatch between number of chunks and embeddings generated."
                }

            # 3. Prepare documents for vector database
            docs_to_add = []
            for i, chunk in enumerate(chunks):
                docs_to_add.append(
                    {
                        "id": f"{processed_document_id}_chunk_{i}",
                        "text": chunk,
                        "embedding": embeddings[i],
                        "metadata": chunk_metadata[i]
                        if i < len(chunk_metadata)
                        else {},
                    }
                )

            # 4. Add documents to vector database
            add_result = self.vector_db_mcp.add_documents(
                documents=docs_to_add, collection_name=collection_name
            )

            if "error" in add_result:
                return {
                    "error": f"Adding documents to vector DB failed: {add_result['error']}"
                }

            # 5. Store document metadata in Redis
            if processed_document_id:
                metadata_key = (
                    f"{self.redis_keys['document_metadata']}{processed_document_id}"
                )
                self.redis_mcp.set_json(
                    metadata_key, document_metadata, ex=self.context_data_ttl
                )

            # Update latest context update timestamp
            self.redis_mcp.set_json(
                self.redis_keys["latest_context_update"],
                {
                    "timestamp": datetime.now().isoformat(),
                    "document_id": processed_document_id,
                },
            )

            return {
                "status": "success",
                "document_id": processed_document_id,
                "chunks_processed": len(chunks),
                "chunks_added_to_vector_db": add_result.get("added_count", 0),
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error processing and storing document: {e}")
            return {"error": str(e)}

    async def retrieve_context(
        self,
        query: str,
        collection_name: str = "default_collection",
        top_k: int = 10,
        domain: Optional[str] = None,
        use_feedback: bool = True,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from the vector database based on a query.

        Args:
            query: The user query
            collection_name: Name of the vector database collection to search
            top_k: Number of top results to retrieve
            domain: Optional domain context for query reformulation
            use_feedback: Whether to use relevance feedback for reranking
            user_id: Optional user ID for personalized feedback

        Returns:
            Dictionary with retrieved context (document chunks and metadata)
        """
        self.logger.info(f"Retrieving context for query: '{query}'")
        start_time = time.time()

        try:
            # 1. Reformulate the query
            reformulated_query_result = self.query_reformulation_mcp.expand_query(
                query=query,
                domain=domain,
                strategies=["synonyms", "domain_terms"],  # Use relevant strategies
            )

            if "error" in reformulated_query_result:
                self.logger.warning(
                    f"Query reformulation failed: {reformulated_query_result['error']}. Using original query."
                )
                search_query = query
            else:
                search_query = reformulated_query_result.get("expanded_query", query)

            # 2. Generate embedding for the search query
            query_embedding_result = self.embeddings_mcp.generate_embeddings(
                chunks=[search_query]
            )

            if "error" in query_embedding_result or not query_embedding_result.get(
                "embeddings"
            ):
                return {
                    "error": f"Query embedding generation failed: {query_embedding_result.get('error', 'Unknown error')}"
                }

            query_embedding = query_embedding_result["embeddings"][0]

            # 3. Search the vector database
            search_results = self.vector_db_mcp.search(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=top_k,
            )

            if "error" in search_results:
                return {
                    "error": f"Vector database search failed: {search_results['error']}"
                }

            retrieved_documents = search_results.get("results", [])

            # 4. Rerank results based on relevance feedback if enabled
            if use_feedback and retrieved_documents:
                # Need a query_id for feedback. Can generate one or use a provided one.
                # For simplicity here, we'll use a hash of the original query.
                query_id = hashlib.md5(query.encode()).hexdigest()

                reranked_results = self.relevance_feedback_mcp.rerank_results(
                    query_id=query_id,
                    results=retrieved_documents,  # Assuming retrieved_documents have 'document_id' and 'score'
                    use_explicit_feedback=True,
                    use_implicit_feedback=True,
                    feedback_weight=0.7,  # Give feedback a higher weight
                )

                if "error" in reranked_results:
                    self.logger.warning(
                        f"Reranking failed: {reranked_results['error']}. Using original search results."
                    )
                    final_results = retrieved_documents
                else:
                    final_results = reranked_results.get(
                        "reranked_results", retrieved_documents
                    )
            else:
                final_results = retrieved_documents

            # 5. Retrieve full document metadata for retrieved chunks
            context_documents = []
            for result in final_results:
                chunk_metadata = result.get("metadata", {})
                document_id = chunk_metadata.get("document_id")
                if document_id:
                    metadata_key = (
                        f"{self.redis_keys['document_metadata']}{document_id}"
                    )
                    document_metadata = self.redis_mcp.get_json(metadata_key) or {}
                    context_documents.append(
                        {
                            "chunk": result.get("text"),
                            "chunk_metadata": chunk_metadata,
                            "document_metadata": document_metadata,
                            "score": result.get("score"),  # Include search/rerank score
                        }
                    )
                else:
                    # Include chunk even if document metadata isn't found
                    context_documents.append(
                        {
                            "chunk": result.get("text"),
                            "chunk_metadata": chunk_metadata,
                            "document_metadata": {},
                            "score": result.get("score"),
                        }
                    )

            # 6. Store query and retrieved context for potential future feedback
            query_history_key = (
                f"{self.redis_keys['query_history']}{user_id or 'anonymous'}"
            )
            query_record = {
                "query": query,
                "search_query": search_query,
                "timestamp": datetime.now().isoformat(),
                "retrieved_document_ids": [
                    res.get("metadata", {}).get("document_id")
                    for res in final_results
                    if res.get("metadata", {}).get("document_id")
                ],
                "results": final_results,  # Store raw results for feedback mapping
            }
            self.redis_mcp.add_to_list(query_history_key, json.dumps(query_record))

            return {
                "original_query": query,
                "search_query": search_query,
                "retrieved_context": context_documents,
                "total_results": len(final_results),
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error retrieving context for query '{query}': {e}")
            return {"error": str(e)}

    async def store_context_data(self, key: str, data: Any) -> Dict[str, Any]:
        """
        Store general contextual data in Redis.

        Args:
            key: The key to store the data under (will be prefixed)
            data: The data to store (JSON serializable)

        Returns:
            Status of the operation
        """
        try:
            full_key = f"{self.redis_keys['context_data']}{key}"
            self.redis_mcp.set_json(full_key, data, ex=self.context_data_ttl)

            # Update latest context update timestamp
            self.redis_mcp.set_json(
                self.redis_keys["latest_context_update"],
                {"timestamp": datetime.now().isoformat(), "key": key},
            )

            return {
                "status": "success",
                "key": key,
                "stored_at": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error storing context data for key '{key}': {e}")
            return {"status": "error", "key": key, "error": str(e)}

    async def get_context_data(self, key: str) -> Dict[str, Any]:
        """
        Retrieve general contextual data from Redis.

        Args:
            key: The key to retrieve the data for (will be prefixed)

        Returns:
            Dictionary with retrieved data or error
        """
        try:
            full_key = f"{self.redis_keys['context_data']}{key}"
            data = self.redis_mcp.get_json(full_key)

            if data is not None:
                return {"status": "success", "key": key, "data": data}
            else:
                return {
                    "status": "not_found",
                    "key": key,
                    "message": "Context data not found for this key",
                }

        except Exception as e:
            self.logger.error(f"Error retrieving context data for key '{key}': {e}")
            return {"status": "error", "key": key, "error": str(e)}

    def run_context_agent(self, query: str) -> Dict[str, Any]:
        """
        Run context management using AutoGen agents.

        Args:
            query: Query or instruction for context management

        Returns:
            Results from the context agent
        """
        self.logger.info(f"Running context agent with query: {query}")

        context_assistant = self.agents.get("context_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not context_assistant or not user_proxy:
            return {"error": "AutoGen agents not initialized"}

        try:
            # Initiate chat with the context assistant
            user_proxy.initiate_chat(context_assistant, message=query)

            # Get the last message from the assistant
            last_message = user_proxy.last_message(context_assistant)
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

            return {"response": content}

        except Exception as e:
            self.logger.error(f"Error during AutoGen chat: {e}")
            return {"error": str(e)}


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
