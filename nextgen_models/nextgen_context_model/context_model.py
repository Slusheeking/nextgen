"""
Context Model

This module defines the ContextModel, responsible for gathering, processing, and managing
contextual information from various sources to support other NextGen models.
It integrates with data retrieval, document analysis, and vector storage MCP tools.
"""

from monitoring.netdata_logger import NetdataLogger
import json
import time
import os
import hashlib
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any, Optional

# For Redis integration (Commented out as RedisMCP is removed)
# from mcp_tools.db_mcp.redis_mcp import RedisMCP
from monitoring.system_metrics import SystemMetricsCollector
from monitoring.stock_charts import StockChartGenerator

# MCP tools
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP # Keep for dynamic loading
from mcp_tools.document_analysis_mcp.document_analysis_mcp import DocumentAnalysisMCP
from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP # Added

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
    reformulate queries, generate embeddings, store/retrieve vectors, and incorporate
    relevance feedback to build a comprehensive understanding of the market and
    relevant information.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ContextModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - data_sources_config: Config for various data source MCPs
                - document_analysis_config: Config for DocumentAnalysisMCP
                - vector_store_config: Config for VectorStoreMCP (ChromaDB)
                - redis_config: Config for RedisMCP (if direct integration is added)
                - context_data_ttl: Time-to-live for context data in seconds (default: 3600 - 1 hour)
                - llm_config: Configuration for AutoGen LLM
        """
        init_start_time = time.time()
        
        # Set up NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-context-model")
        self.logger.info("Context Model initialization started")
        
        # Initialize counters for performance monitoring
        self.document_processing_count = 0
        self.embedding_generation_count = 0
        self.vector_store_operations = 0
        self.vector_search_count = 0
        self.query_expansion_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0

        # Initialize system metrics collector
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.logger.info("System metrics collector initialized")

        # Initialize chart generator for financial data visualization
        self.chart_generator = StockChartGenerator()
        self.logger.info("Stock chart generator initialized")

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
        # Data Sources
        self.data_sources: Dict[str, BaseDataMCP] = {}
        data_sources_config = self.config.get("data_sources_config", {})
        for source_name, source_config in data_sources_config.items():
            try:
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

        # Document Analysis MCP
        self.document_analysis_mcp = DocumentAnalysisMCP(
            self.config.get("document_analysis_config")
        )
        self.logger.info("Initialized DocumentAnalysisMCP")

        # Vector Store MCP (ChromaDB)
        self.vector_store_mcp = VectorStoreMCP(
            self.config.get("vector_store_config")
        )
        self.logger.info("Initialized VectorStoreMCP")


        # Redis client placeholder (if direct integration is needed)
        self.redis_client = None

        # Configuration parameters
        self.context_data_ttl = self.config.get(
            "context_data_ttl", 3600
        )  # Default: 1 hour

        # Redis keys for data storage (Commented out - requires Redis integration)
        self.redis_keys = {} # Placeholder

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
            2. Process and structure raw information for easy retrieval and analysis (chunking, metadata extraction)
            3. Generate embeddings for text chunks and store them in a vector database
            4. Retrieve relevant documents from the vector database based on semantic similarity
            5. Reformulate user queries to improve search results
            6. Incorporate user feedback to refine context and retrieval

            You have tools for data retrieval, document analysis (processing, embeddings, query reformulation, feedback),
            and vector store operations (add, search). Always aim to provide the most relevant and accurate context.""",
            llm_config=self.llm_config,
            description="A specialist in gathering, processing, storing, and retrieving financial market context",
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

        # Register data retrieval functions
        async def fetch_data_func(source_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
            return await self.fetch_data(source_name, query)
            
        register_function(
            fetch_data_func,
            name="fetch_data",
            description="Fetch data from a specified data source",
            caller=context_assistant,
            executor=user_proxy,
        )

        # Register document processing functions (using DocumentAnalysisMCP)
        async def process_document_func(
            document: str,
            document_id: Optional[str] = None,
            document_type: Optional[str] = None,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            chunk_strategy: str = "paragraph",
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "process_document", # Assuming tool name remains the same
                {
                    "document": document,
                    "document_id": document_id,
                    "document_type": document_type,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunk_strategy": chunk_strategy,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Document processing failed"}
            
        register_function(
            process_document_func,
            name="process_document",
            description="Process a document (clean, extract metadata, and chunk)",
            caller=context_assistant,
            executor=user_proxy,
        )

        async def deduplicate_chunks_func(
            chunks: List[str],
            similarity_threshold: float = 0.9,
            method: str = "jaccard",
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "deduplicate_chunks", # Assuming tool name remains the same
                {"chunks": chunks, "similarity_threshold": similarity_threshold, "method": method}
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Chunk deduplication failed"}
            
        register_function(
            deduplicate_chunks_func,
            name="deduplicate_chunks",
            description="Remove duplicate or near-duplicate chunks",
            caller=context_assistant,
            executor=user_proxy,
        )

        async def generate_embeddings_func(
            chunks: List[str], model: Optional[str] = None
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "generate_embeddings", # Assuming tool name remains the same
                {"chunks": chunks, "model": model}
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Embedding generation failed"}
            
        register_function(
            generate_embeddings_func,
            name="generate_embeddings",
            description="Generate embeddings for text chunks",
            caller=context_assistant,
            executor=user_proxy,
        )

        # Register vector database functions (using VectorStoreMCP)
        async def add_documents_to_vector_db_func(
            documents: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            ids: Optional[List[str]] = None,
            collection_name: Optional[str] = None,
        ) -> Dict[str, Any]:
            self.mcp_tool_call_count += 1
            result = self.vector_store_mcp.call_tool(
                "add_documents",
                {
                    "documents": documents,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "ids": ids,
                    "collection_name": collection_name,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Adding documents to vector DB failed"}
            
        register_function(
            add_documents_to_vector_db_func,
            name="add_documents_to_vector_db",
            description="Add documents (chunks, embeddings, metadata) to the vector database",
            caller=context_assistant,
            executor=user_proxy,
        )

        async def search_vector_db_func(
            query_embeddings: List[List[float]],
            collection_name: Optional[str] = None,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            include: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            self.mcp_tool_call_count += 1
            result = self.vector_store_mcp.call_tool(
                "search_collection",
                {
                    "query_embeddings": query_embeddings,
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "filters": filters,
                    "include": include,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Vector DB search failed"}
            
        register_function(
            search_vector_db_func,
            name="search_vector_db",
            description="Search the vector database using query embeddings",
            caller=context_assistant,
            executor=user_proxy,
        )

        # Register query reformulation functions (using DocumentAnalysisMCP)
        async def expand_query_func(
            query: str,
            domain: Optional[str] = None,
            strategies: Optional[List[str]] = None,
            max_terms: Optional[int] = None,
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "expand_query", # Assuming tool name remains the same
                {"query": query, "domain": domain, "strategies": strategies, "max_terms": max_terms}
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Query expansion failed"}
            
        register_function(
            expand_query_func,
            name="expand_query",
            description="Expand a query with related terms to improve retrieval",
            caller=context_assistant,
            executor=user_proxy,
        )

        async def decompose_query_func(
            query: str, max_sub_queries: Optional[int] = None
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "decompose_query", # Assuming tool name remains the same
                {"query": query, "max_sub_queries": max_sub_queries}
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Query decomposition failed"}
            
        register_function(
            decompose_query_func,
            name="decompose_query",
            description="Break down a complex query into simpler sub-queries",
            caller=context_assistant,
            executor=user_proxy,
        )

        # Register relevance feedback functions (using DocumentAnalysisMCP)
        async def record_explicit_feedback_func(
            query_id: str,
            document_id: str,
            rating: float,
            query_text: Optional[str] = None,
            document_text: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "record_explicit_feedback", # Assuming tool name remains the same
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "rating": rating,
                    "query_text": query_text,
                    "document_text": document_text,
                    "user_id": user_id,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Recording explicit feedback failed"}
            
        register_function(
            record_explicit_feedback_func,
            name="record_explicit_feedback",
            description="Record explicit user feedback on document relevance",
            caller=context_assistant,
            executor=user_proxy,
        )

        async def record_implicit_feedback_func(
            query_id: str,
            document_id: str,
            interaction_type: str,
            interaction_value: float,
            query_text: Optional[str] = None,
            document_text: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "record_implicit_feedback", # Assuming tool name remains the same
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "interaction_type": interaction_type,
                    "interaction_value": interaction_value,
                    "query_text": query_text,
                    "document_text": document_text,
                    "user_id": user_id,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Recording implicit feedback failed"}
            
        register_function(
            record_implicit_feedback_func,
            name="record_implicit_feedback",
            description="Record implicit user feedback based on interactions",
            caller=context_assistant,
            executor=user_proxy,
        )

        async def rerank_results_func(
            query_id: str,
            results: List[Dict[str, Any]],
            use_explicit_feedback: bool = True,
            use_implicit_feedback: bool = True,
            feedback_weight: Optional[float] = None,
        ) -> Dict[str, Any]:
            # Call the consolidated tool
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(
                "rerank_results", # Assuming tool name remains the same
                {
                    "query_id": query_id,
                    "results": results,
                    "use_explicit_feedback": use_explicit_feedback,
                    "use_implicit_feedback": use_implicit_feedback,
                    "feedback_weight": feedback_weight,
                }
            )
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result or {"error": "Reranking results failed"}
            
        register_function(
            rerank_results_func,
            name="rerank_results",
            description="Rerank search results based on feedback",
            caller=context_assistant,
            executor=user_proxy,
        )

        # Register data storage and retrieval functions (Commented out - requires Redis)
        # ...

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
            # Create a function with a closure over data_source_mcp
            def create_data_source_tool_func(mcp_client):
                def use_data_source_tool_func(tool_name: str, arguments: Dict[str, Any]) -> Any:
                    self.mcp_tool_call_count += 1
                    result = mcp_client.call_tool(tool_name, arguments)
                    if result and result.get("error"):
                        self.mcp_tool_error_count += 1
                    return result
                return use_data_source_tool_func
            
            # Create the function with the closure
            data_source_tool_func = create_data_source_tool_func(data_source_mcp)
            
            # Register the function
            register_function(
                data_source_tool_func,
                name=f"use_{source_name}_tool",
                description=f"Use a tool provided by the {source_name} MCP server",
                caller=context_assistant,
                executor=user_proxy,
            )

        # Define MCP tool access functions for analysis tools
        def use_document_analysis_tool_func(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.document_analysis_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result
            
        register_function(
            use_document_analysis_tool_func,
            name="use_document_analysis_tool",
            description="Use a tool provided by the Document Analysis MCP server",
            caller=context_assistant,
            executor=user_proxy,
        )

        # Define MCP tool access function for Vector Store
        def use_vector_store_tool_func(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.vector_store_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result
            
        register_function(
            use_vector_store_tool_func,
            name="use_vector_store_tool",
            description="Use a tool provided by the Vector Store MCP server (ChromaDB)",
            caller=context_assistant,
            executor=user_proxy,
        )


    async def fetch_data(
        self, source_name: str, query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fetch data from a specified data source MCP.
        (Code unchanged from previous version)
        """
        try:
            if source_name not in self.data_sources:
                return {
                    "error": f"Data source '{source_name}' not initialized or supported."
                }

            data_source_mcp = self.data_sources[source_name]

            if hasattr(data_source_mcp, "fetch"):
                self.mcp_tool_call_count += 1
                result = data_source_mcp.fetch(**query)
                if result and result.get("error"): self.mcp_tool_error_count += 1
                return result
            elif hasattr(data_source_mcp, "call_tool"):
                tool_name = query.get("tool_name")
                tool_args = query.get("args", {})
                if tool_name:
                    self.mcp_tool_call_count += 1
                    result = data_source_mcp.call_tool(tool_name, tool_args)
                    if result and result.get("error"): self.mcp_tool_error_count += 1
                    return result
                else:
                    return {"error": f"No tool_name specified for data source '{source_name}'."}
            else:
                return {"error": f"Data source '{source_name}' does not have a 'fetch' or 'call_tool' method."}

        except Exception as e:
            self.logger.error(f"Error fetching data from {source_name}: {e}")
            return {"error": str(e)}

    async def process_and_store_document(
        self,
        document: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        collection_name: Optional[str] = None, # Use VectorStoreMCP default if None
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_strategy: str = "paragraph",
    ) -> Dict[str, Any]:
        """
        Process a document, generate embeddings, and store in the vector database (ChromaDB).

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
            f"Processing and storing document (ID: {document_id or 'auto-generated'}) in collection '{collection_name or 'default'}'", 
            document_type=document_type,
            chunk_strategy=chunk_strategy
        )
        start_time = time.time()
        self.document_processing_count += 1
        self.logger.counter("context_model.document_processing_count")

        try:
            # 1. Process the document (clean, metadata, chunk) using DocumentAnalysisMCP
            processing_start = time.time()
            self.mcp_tool_call_count += 1
            process_result = self.document_analysis_mcp.call_tool(
                "process_document",
                {
                    "document": document,
                    "document_id": document_id,
                    "document_type": document_type,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunk_strategy": chunk_strategy,
                }
            )
            
            processing_duration = (time.time() - processing_start) * 1000
            self.logger.timing("context_model.document_processing_time_ms", processing_duration)

            if not process_result or process_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.counter("context_model.mcp_tool_error_count")
                 self.execution_errors += 1
                 self.logger.counter("context_model.execution_errors")
                 error_msg = process_result.get("error", "Unknown processing error") if process_result else "No processing result"
                 self.logger.error(f"Document processing failed: {error_msg}", 
                                  document_type=document_type,
                                  document_id=document_id)
                 return {"error": f"Document processing failed: {error_msg}"}

            chunks = process_result.get("chunks", [])
            chunk_metadata = process_result.get("chunk_metadata", []) # Metadata per chunk
            document_metadata = process_result.get("document_metadata", {}) # Overall document metadata
            processed_document_id = process_result.get("document_id") # ID generated/used by processing
            
            self.logger.gauge("context_model.chunks_per_document", len(chunks))
            self.logger.info("Document processed successfully", 
                           document_id=processed_document_id, 
                           chunk_count=len(chunks))

            if not chunks:
                self.logger.warning("No chunks generated from document", document_id=processed_document_id)
                return {"status": "skipped", "message": "No chunks generated from document."}

            # 2. Generate embeddings for chunks (using DocumentAnalysisMCP)
            embedding_start = time.time()
            self.mcp_tool_call_count += 1
            self.embedding_generation_count += 1
            self.logger.counter("context_model.embedding_generation_count")
            
            embedding_result = self.document_analysis_mcp.call_tool(
                "generate_embeddings",
                {"chunks": chunks}
            )
            
            embedding_duration = (time.time() - embedding_start) * 1000
            self.logger.timing("context_model.embedding_generation_time_ms", embedding_duration)

            if not embedding_result or embedding_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.counter("context_model.mcp_tool_error_count")
                 self.execution_errors += 1
                 self.logger.counter("context_model.execution_errors")
                 error_msg = embedding_result.get("error", "Unknown embedding error") if embedding_result else "No embedding result"
                 self.logger.error(f"Embedding generation failed: {error_msg}", document_id=processed_document_id)
                 return {"error": f"Embedding generation failed: {error_msg}"}

            embeddings = embedding_result.get("embeddings", [])
            
            self.logger.info(f"Generated embeddings for {len(embeddings)} chunks", 
                           document_id=processed_document_id)

            if len(chunks) != len(embeddings):
                 self.execution_errors += 1
                 self.logger.counter("context_model.execution_errors")
                 self.logger.error("Mismatch between number of chunks and embeddings", 
                                  chunks=len(chunks),
                                  embeddings=len(embeddings),
                                  document_id=processed_document_id)
                 return {"error": "Mismatch between number of chunks and embeddings generated."}

            # 3. Prepare documents for vector database (ChromaDB format)
            # ChromaDB needs lists of embeddings, documents, metadatas, and ids
            ids_for_db = [f"{processed_document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas_for_db = []
            for i, chunk_meta in enumerate(chunk_metadata):
                # Combine chunk-specific metadata with overall document metadata
                combined_meta = document_metadata.copy()
                combined_meta.update(chunk_meta)
                combined_meta["document_id"] = processed_document_id # Ensure doc ID is in metadata
                combined_meta["chunk_index"] = i
                metadatas_for_db.append(combined_meta)


            # 4. Add documents to vector database (using VectorStoreMCP)
            db_start = time.time()
            self.mcp_tool_call_count += 1
            self.vector_store_operations += 1
            self.logger.counter("context_model.vector_store_operations")
            
            add_result = self.vector_store_mcp.call_tool(
                "add_documents",
                {
                    "documents": chunks,
                    "embeddings": embeddings,
                    "metadatas": metadatas_for_db,
                    "ids": ids_for_db,
                    "collection_name": collection_name # Pass along specified collection name
                }
            )
            
            db_duration = (time.time() - db_start) * 1000
            self.logger.timing("context_model.vector_db_operation_time_ms", db_duration)

            if not add_result or add_result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.counter("context_model.mcp_tool_error_count")
                self.execution_errors += 1
                self.logger.counter("context_model.execution_errors")
                error_msg = add_result.get("error", "Unknown DB error") if add_result else "No DB result"
                self.logger.error(f"Adding documents to vector DB failed: {error_msg}", 
                                 document_id=processed_document_id,
                                 collection=collection_name)
                return {"error": f"Adding documents to vector DB failed: {error_msg}"}

            # 5. Store document metadata in Redis (Commented out)
            # ...

            # Update latest context update timestamp (Commented out)
            # ...
            
            total_duration = time.time() - start_time
            self.logger.info("Document successfully processed and stored in vector database", 
                           document_id=processed_document_id,
                           chunks_added=add_result.get("added_count", 0),
                           processing_time_ms=total_duration * 1000,
                           collection=add_result.get("collection_name", "unknown"))
            
            self.logger.gauge("context_model.document_processing_success", 1)
            self.logger.timing("context_model.total_document_processing_time_ms", total_duration * 1000)

            return {
                "status": "success",
                "document_id": processed_document_id,
                "chunks_processed": len(chunks),
                "chunks_added_to_vector_db": add_result.get("added_count", 0),
                "collection_name": add_result.get("collection_name"),
                "processing_time": total_duration,
            }

        except Exception as e:
            self.execution_errors += 1
            self.logger.counter("context_model.execution_errors")
            self.logger.error(f"Error processing and storing document: {e}", 
                             document_id=document_id,
                             document_type=document_type,
                             exc_info=True)
            self.logger.gauge("context_model.document_processing_failure", 1)
            return {"error": str(e)}

    async def retrieve_context(
        self,
        query: str,
        collection_name: Optional[str] = None, # Use VectorStoreMCP default if None
        top_k: int = 10,
        domain: Optional[str] = None,
        use_feedback: bool = True, # Controls reranking step
        user_id: Optional[str] = None, # For feedback history
        filters: Optional[Dict[str, Any]] = None, # Metadata filters for ChromaDB
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from the vector database (ChromaDB) based on a query.

        Args:
            query: The user query
            collection_name: Name of the vector database collection to search
            top_k: Number of top results to retrieve
            domain: Optional domain context for query reformulation
            use_feedback: Whether to use relevance feedback for reranking
            user_id: Optional user ID for personalized feedback
            filters: Optional metadata filters for the vector search

        Returns:
            Dictionary with retrieved context (document chunks and metadata)
        """
        self.logger.info(f"Retrieving context for query: '{query}'", 
                        collection=collection_name or 'default',
                        domain=domain,
                        top_k=top_k,
                        use_feedback=use_feedback)
        
        start_time = time.time()
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]  # Short hash for logging
        self.vector_search_count += 1
        self.logger.counter("context_model.vector_search_count")

        try:
            # 1. Reformulate the query (using DocumentAnalysisMCP)
            query_expansion_start = time.time()
            self.mcp_tool_call_count += 1
            self.query_expansion_count += 1
            self.logger.counter("context_model.query_expansion_count")
            
            reformulated_query_result = self.document_analysis_mcp.call_tool(
                "expand_query",
                {"query": query, "domain": domain, "strategies": ["synonyms", "domain_terms"]}
            )
            
            query_expansion_duration = (time.time() - query_expansion_start) * 1000
            self.logger.timing("context_model.query_expansion_time_ms", query_expansion_duration)

            if reformulated_query_result and reformulated_query_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.counter("context_model.mcp_tool_error_count")
                 self.logger.warning(
                     f"Query reformulation failed: {reformulated_query_result['error']}. Using original query.",
                     query_hash=query_hash,
                     domain=domain
                 )
                 search_query = query
            else:
                 search_query = reformulated_query_result.get("expanded_query", query) if reformulated_query_result else query
                 if search_query != query:
                     self.logger.info("Query expanded successfully", 
                                    original_query=query,
                                    expanded_query=search_query,
                                    query_hash=query_hash)

            # 2. Generate embedding for the search query (using DocumentAnalysisMCP)
            embedding_start = time.time()
            self.mcp_tool_call_count += 1
            self.embedding_generation_count += 1
            self.logger.counter("context_model.embedding_generation_count")
            
            query_embedding_result = self.document_analysis_mcp.call_tool(
                "generate_embeddings",
                {"chunks": [search_query]}
            )
            
            embedding_duration = (time.time() - embedding_start) * 1000
            self.logger.timing("context_model.query_embedding_time_ms", embedding_duration)

            if not query_embedding_result or query_embedding_result.get("error") or not query_embedding_result.get("embeddings"):
                self.mcp_tool_error_count += 1
                self.logger.counter("context_model.mcp_tool_error_count")
                self.execution_errors += 1
                self.logger.counter("context_model.execution_errors")
                error_msg = query_embedding_result.get("error", "Unknown embedding error") if query_embedding_result else "No embedding result"
                self.logger.error(f"Query embedding generation failed: {error_msg}", 
                                 query_hash=query_hash)
                return {"error": f"Query embedding generation failed: {error_msg}"}

            query_embedding = query_embedding_result["embeddings"][0] # Get the single embedding
            self.logger.info("Query embedding generated", query_hash=query_hash)

            # 3. Search the vector database (using VectorStoreMCP)
            search_start = time.time()
            self.mcp_tool_call_count += 1
            self.vector_store_operations += 1
            self.logger.counter("context_model.vector_store_operations")
            
            search_results_mcp = self.vector_store_mcp.call_tool(
                "search_collection",
                {
                    "query_embeddings": [query_embedding], # Search tool expects a list of embeddings
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "filters": filters,
                    "include": ["metadatas", "documents", "distances"] # Request necessary fields
                }
            )
            
            search_duration = (time.time() - search_start) * 1000
            self.logger.timing("context_model.vector_search_time_ms", search_duration)

            if not search_results_mcp or search_results_mcp.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.counter("context_model.mcp_tool_error_count")
                self.execution_errors += 1
                self.logger.counter("context_model.execution_errors")
                error_msg = search_results_mcp.get("error", "Unknown DB error") if search_results_mcp else "No DB result"
                self.logger.error(f"Vector database search failed: {error_msg}", 
                                 query_hash=query_hash,
                                 collection=collection_name)
                return {"error": f"Vector database search failed: {error_msg}"}

            # Extract results from the MCP response structure
            # ChromaDB query returns lists for each field (ids, documents, metadatas, distances)
            # Since we sent one query embedding, we expect results at index 0
            search_results_raw = search_results_mcp.get("results", {})
            retrieved_ids = search_results_raw.get("ids", [[]])[0]
            retrieved_docs = search_results_raw.get("documents", [[]])[0]
            retrieved_metadatas = search_results_raw.get("metadatas", [[]])[0]
            retrieved_distances = search_results_raw.get("distances", [[]])[0]
            
            result_count = len(retrieved_ids)
            self.logger.info(f"Vector search retrieved {result_count} results", 
                           query_hash=query_hash,
                           collection=collection_name,
                           filter_applied=filters is not None)
            self.logger.gauge("context_model.search_result_count", result_count)

            # Combine into a list of dictionaries for easier processing
            initial_results = []
            for i in range(len(retrieved_ids)):
                initial_results.append({
                    "id": retrieved_ids[i],
                    "text": retrieved_docs[i],
                    "metadata": retrieved_metadatas[i],
                    "distance": retrieved_distances[i],
                    "score": 1 - retrieved_distances[i] # Convert distance to similarity score (optional)
                })
                
            # Calculate average relevance score for metrics
            if initial_results:
                avg_score = sum(result["score"] for result in initial_results) / len(initial_results)
                self.logger.gauge("context_model.avg_result_relevance", avg_score)

            # 4. Rerank results based on relevance feedback if enabled (using DocumentAnalysisMCP)
            final_results = initial_results
            if use_feedback and initial_results:
                rerank_start = time.time()
                query_id = hashlib.md5(query.encode()).hexdigest() # Simple query ID
                
                self.mcp_tool_call_count += 1
                reranked_results_mcp = self.document_analysis_mcp.call_tool(
                    "rerank_results",
                     {
                         "query_id": query_id,
                         "results": initial_results, # Pass combined results
                         "use_explicit_feedback": True,
                         "use_implicit_feedback": True,
                         # feedback_weight: Optional[float] = None # Use default or configure
                     }
                )
                
                rerank_duration = (time.time() - rerank_start) * 1000
                self.logger.timing("context_model.reranking_time_ms", rerank_duration)

                if not reranked_results_mcp or reranked_results_mcp.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.counter("context_model.mcp_tool_error_count")
                    self.logger.warning(
                        f"Reranking failed: {reranked_results_mcp.get('error', 'Unknown reranking error')}. Using original search results.",
                        query_hash=query_hash
                    )
                else:
                    final_results = reranked_results_mcp.get("reranked_results", initial_results)
                    self.logger.info("Results reranked successfully", 
                                   query_hash=query_hash,
                                   reranking_used=True)
                    
                    # Track if reranking changed the order significantly
                    if final_results and initial_results:
                        # Check if top result changed
                        top_changed = final_results[0]["id"] != initial_results[0]["id"] if final_results and initial_results else False
                        self.logger.gauge("context_model.reranking_changed_top_result", 1 if top_changed else 0)

            # 5. Format context documents (already done during combination)
            context_documents = final_results

            # 6. Store query history (Placeholder - requires Redis)
            # ...
            
            total_duration = time.time() - start_time
            self.logger.info("Context retrieval completed successfully", 
                           query_hash=query_hash,
                           result_count=len(context_documents),
                           total_time_ms=total_duration * 1000)
            
            # Record metrics for this retrieval operation
            self.logger.timing("context_model.total_retrieval_time_ms", total_duration * 1000)
            self.logger.gauge("context_model.retrieval_success", 1)
            
            # If we have a multi-symbol financial query, we could generate a comparison chart
            if domain == "finance" and len(context_documents) > 0:
                # Extract potential stock symbols from the query
                # This is just a simplistic example - in a real system you'd want more sophisticated extraction
                potential_symbols = [word.upper() for word in query.split() 
                                   if word.isalpha() and len(word) <= 5 and word.isupper()]
                
                if len(potential_symbols) >= 2:
                    try:
                        # Generate a comparison chart if multiple symbols are detected
                        chart_file = self.chart_generator.create_multi_stock_chart(potential_symbols, normalize=True)
                        self.logger.info("Generated comparison chart for symbols in query", 
                                       symbols=potential_symbols,
                                       chart_file=chart_file)
                    except Exception as chart_e:
                        self.logger.warning(f"Failed to generate chart for query: {chart_e}")

            return {
                "original_query": query,
                "search_query": search_query,
                "retrieved_context": context_documents,
                "total_results": len(context_documents),
                "processing_time": total_duration,
            }

        except Exception as e:
            self.execution_errors += 1
            self.logger.counter("context_model.execution_errors")
            self.logger.error(f"Error retrieving context for query: {e}", 
                             query=query,
                             query_hash=query_hash,
                             exc_info=True)
            self.logger.gauge("context_model.retrieval_failure", 1)
            return {"error": str(e)}
            
    def retrieve_context_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Synchronous wrapper for retrieve_context that handles the async coroutine.
        
        Args:
            query: The user query
            **kwargs: Additional keyword arguments to pass to retrieve_context
            
        Returns:
            Dictionary with retrieved context or error information
        """
        import asyncio
        try:
            # Create a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # No event loop in this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run the async method in the event loop
            return loop.run_until_complete(self.retrieve_context(query, **kwargs))
        except Exception as e:
            self.logger.error(f"Error in synchronous wrapper for retrieve_context: {e}")
            return {"error": f"Synchronous wrapper error: {str(e)}"}

    async def store_context_data(self, key: str, data: Any) -> Dict[str, Any]:
        """
        Store general contextual data.
        (Placeholder - Redis integration removed)
        """
        self.logger.warning(f"Attempted to store context data for key '{key}', but Redis storage is not available.")
        return {"status": "error", "key": key, "error": "Redis storage integration is not available"}


    async def get_context_data(self, key: str) -> Dict[str, Any]:
        """
        Retrieve general contextual data.
        (Placeholder - Redis integration removed)
        """
        self.logger.warning(f"Attempted to get context data for key '{key}', but Redis storage is not available.")
        return {"status": "error", "key": key, "error": "Redis storage integration is not available"}


    def run_context_agent(self, query: str) -> Dict[str, Any]:
        """
        Run context management using AutoGen agents.
        (Code unchanged from previous version)
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


    def shutdown(self):
        """
        Shutdown the context model, stopping all monitoring systems gracefully.
        This should be called when the model is no longer needed to ensure proper cleanup.
        """
        self.logger.info("Shutting down Context Model")
        
        # Stop the system metrics collector
        if hasattr(self, 'metrics_collector'):
            try:
                self.metrics_collector.stop()
                self.logger.info("System metrics collection stopped")
            except Exception as e:
                self.logger.error(f"Error stopping metrics collector: {e}")
        
        # Log final metrics before shutdown
        self.logger.info("Context Model shutdown complete", 
                        document_processing_count=self.document_processing_count,
                        embedding_generation_count=self.embedding_generation_count,
                        vector_store_operations=self.vector_store_operations,
                        vector_search_count=self.vector_search_count,
                        query_expansion_count=self.query_expansion_count,
                        mcp_tool_error_count=self.mcp_tool_error_count,
                        execution_errors=self.execution_errors)
        
        # Record total uptime
        self.logger.gauge("context_model.successful_shutdown", 1)
        self.logger.info("Context Model resources released")


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
