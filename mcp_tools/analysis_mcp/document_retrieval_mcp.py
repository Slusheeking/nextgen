"""
Document Retrieval MCP Tool

This module implements a Model Context Protocol (MCP) server for retrieving
relevant documents or text snippets based on semantic similarity using
vector embeddings and a vector database. It assumes interaction with
EmbeddingsMCP and VectorDBMCP through appropriate client mechanisms
or direct calls if running within the same orchestrated environment.
"""

import os
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Import necessary libraries for potential reranking
try:
    from sentence_transformers.cross_encoder import CrossEncoder

    HAVE_CROSS_ENCODER = True
except ImportError:
    HAVE_CROSS_ENCODER = False
    CrossEncoder = None  # Define for type hinting

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-document-retrieval",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class DocumentRetrievalMCP(BaseMCPServer):
    """
    MCP server for retrieving documents using semantic search.

    Leverages vector embeddings and a vector database to find documents
    or text passages relevant to a given query. Requires access to
    EmbeddingsMCP and VectorDBMCP services or equivalent functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Document Retrieval MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - vector_db_mcp_client: Client or config to interact with VectorDBMCP.
                - embeddings_mcp_client: Client or config to interact with EmbeddingsMCP.
                - default_collection: Default collection name in the vector DB.
                - default_top_k: Default number of results to retrieve.
                - reranker_model_name: Name of a cross-encoder model for reranking.
                - reranker_cache_dir: Cache directory for the reranker model.
        """
        super().__init__(name="document_retrieval_mcp", config=config)

        # Configurations
        # These would ideally be client instances or connection details
        self.vector_db_mcp_client = self.config.get("vector_db_mcp_client")
        self.embeddings_mcp_client = self.config.get("embeddings_mcp_client")
        self.default_collection_name = self.config.get(
            "default_collection", "financial_context"
        )
        self.default_top_k = self.config.get("default_top_k", 5)
        self.reranker_model_name = self.config.get("reranker_model_name")
        self.reranker_cache_dir = self.config.get(
            "reranker_cache_dir", "./reranker_models_cache"
        )

        # Validate dependent service configurations/clients
        if not self.vector_db_mcp_client:
            self.logger.error("VectorDBMCP client/config is required but not provided.")
            raise ValueError(
                "VectorDBMCP client/config must be provided in the configuration."
            )
        if not self.embeddings_mcp_client:
            self.logger.warning(
                "EmbeddingsMCP client/config not provided. Retrieval by text query will not work."
            )
            #
            # Allow initialization but log warning, as retrieval by vector
            # might still work

        # Load reranker model if specified
        self.reranker: Any = None  # type: ignore  # CrossEncoder may be None if import fails
        self._load_reranker_model()

        # Register tools
        self._register_tools()

    def _load_reranker_model(self):
        """Load a cross-encoder model for reranking search results."""
        if self.reranker_model_name:
            if not HAVE_CROSS_ENCODER:
                self.logger.warning(
                    "Sentence Transformers library (for CrossEncoder) not found, but reranker model specified. Reranking disabled."
                )
                return

            try:
                # Ensure cache directory exists
                if self.reranker_cache_dir and not os.path.exists(
                    self.reranker_cache_dir
                ):
                    try:
                        os.makedirs(self.reranker_cache_dir, exist_ok=True)
                    except OSError as e:
                        self.logger.error(
                            f"Failed to create reranker cache directory {self.reranker_cache_dir}: {e}"
                        )
                        self.reranker_cache_dir = (
                            None  # Disable caching if dir creation fails
                        )

                self.logger.info(f"Loading reranker model: {self.reranker_model_name}")
                self.reranker = CrossEncoder(
                    self.reranker_model_name,
                    cache_folder=self.reranker_cache_dir
                    if self.reranker_cache_dir
                    else None,
                )
                # Move reranker to CUDA if available and requested
                use_gpu = getattr(self, "use_gpu", True)
                device = (
                    "cuda"
                    if use_gpu
                    and hasattr(self, "_is_cuda_available")
                    and self._is_cuda_available()
                    else "cpu"
                )
                if device == "cuda":
                    try:
                        self.reranker = self.reranker.to("cuda")
                    except Exception as e:
                        self.logger.warning(
                            f"Could not move reranker model to CUDA: {e}"
                        )
                self.logger.info(
                    f"Successfully loaded reranker model: {self.reranker_model_name}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load reranker model '{self.reranker_model_name}': {e}"
                )
                self.reranker = None
        else:
            self.logger.info(
                "No reranker model specified. Using vector similarity scores only."
            )

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "retrieve_by_text",
            self.retrieve_by_text,
            "Retrieve relevant documents/snippets based on a text query using semantic search",
            {
                "query_text": {
                    "type": "string",
                    "description": "The text query to search for relevant documents",
                },
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection to search (defaults to default collection)",
                    "required": False,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of initial results to retrieve from vector DB",
                    "default": self.default_top_k
                    * 2,  # Retrieve more initially if reranking
                },
                "final_top_k": {
                    "type": "integer",
                    "description": "Number of final results to return after potential reranking",
                    "default": self.default_top_k,
                },
                "where_filter": {
                    "type": "object",
                    "description": "Filter conditions for metadata (specific to vector DB)",
                    "required": False,
                },
                "use_reranker": {
                    "type": "boolean",
                    "description": "Whether to use a cross-encoder model to rerank results (if available)",
                    "default": True,  # Default to true if reranker is loaded
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include metadata in the results",
                    "default": True,
                },
                "include_text": {
                    "type": "boolean",
                    "description": "Whether to include the document text/snippet in the results (if stored)",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {"type": "object"},
                    },  # List of {id, score, metadata?, text_snippet?}
                    "collection_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

        self.register_tool(
            "retrieve_by_vector",
            self.retrieve_by_vector,
            "Retrieve relevant documents/snippets based on a query vector",
            {
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "The query vector embedding",
                },
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection to search (defaults to default collection)",
                    "required": False,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to retrieve",
                    "default": self.default_top_k,
                },
                "where_filter": {
                    "type": "object",
                    "description": "Filter conditions for metadata (specific to vector DB)",
                    "required": False,
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include metadata in the results",
                    "default": True,
                },
                "include_text": {
                    "type": "boolean",
                    "description": "Whether to include the document text/snippet in the results (if stored)",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {"type": "object"},
                    },  # List of {id, score, metadata?, text_snippet?}
                    "collection_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

    async def _call_embedding_service(self, text: str) -> Optional[List[float]]:
        """Helper to call the embedding service (simulated)."""
        if not self.embeddings_mcp_client:
            self.logger.error(
                "Cannot generate embedding: EmbeddingsMCP client/config not available."
            )
            return None

        # In a real system, this would make an RPC/API call to EmbeddingsMCP
        # Example using a hypothetical client:
        #
        # result = await
        # self.embeddings_mcp_client.generate_embedding(text=text,
        # normalize=True)
        # if result and not result.get("error"):
        #     return result.get("embedding")
        # else:
        #
        # self.logger.error(f"Failed to get embedding: {result.get('error')}")
        #     return None

        # --- Simulation for local testing ---
        #
        # This part simulates the call if EmbeddingsMCP was instantiated
        # directly
        # It should be replaced by actual client interaction in production.
        try:
            #
            # Attempt direct call if possible (not recommended for distributed
            # MCP)
            if hasattr(self.embeddings_mcp_client, "generate_embedding"):
                result = self.embeddings_mcp_client.generate_embedding(
                    text=text, normalize=True
                )
                if result and not result.get("error"):
                    return result.get("embedding")
                else:
                    self.logger.error(
                        f"Simulated embedding call failed: {result.get('error')}"
                    )
                    return None
            else:
                self.logger.error(
                    "Simulated embedding client does not have 'generate_embedding' method."
                )
                return None
        except Exception as e:
            self.logger.error(f"Error during simulated embedding call: {e}")
            return None
        # --- End Simulation ---

    async def _call_vector_db_query(
        self,
        collection_name: str,
        query_vector: List[List[float]],
        top_k: int,
        where_filter: Optional[Dict],
        include: List[str],
    ) -> Optional[Dict]:
        """Helper to call the vector DB query service (simulated)."""
        if not self.vector_db_mcp_client:
            self.logger.error(
                "Cannot query vector DB: VectorDBMCP client/config not available."
            )
            return None

        # In a real system, make RPC/API call to VectorDBMCP
        # Example:
        # result = await self.vector_db_mcp_client.query_similar(
        #     collection_name=collection_name,
        #     query_embeddings=query_vector,
        #     n_results=top_k,
        #     where_filter=where_filter,
        #     include=include
        # )
        # if result and not result.get("error"):
        #     return result.get("results")
        # else:
        #
        # self.logger.error(f"Failed to query vector DB:
        # {result.get('error')}")
        #     return None

        # --- Simulation for local testing ---
        try:
            if hasattr(self.vector_db_mcp_client, "query_similar"):
                result = self.vector_db_mcp_client.query_similar(
                    collection_name=collection_name,
                    query_embeddings=query_vector,
                    n_results=top_k,
                    where_filter=where_filter,
                    include=include,
                )
                if result and not result.get("error"):
                    return result.get("results")
                else:
                    self.logger.error(
                        f"Simulated vector DB query failed: {result.get('error')}"
                    )
                    return None
            else:
                self.logger.error(
                    "Simulated vector DB client does not have 'query_similar' method."
                )
                return None
        except Exception as e:
            self.logger.error(f"Error during simulated vector DB query: {e}")
            return None
        # --- End Simulation ---

    async def _call_vector_db_get(
        self, collection_name: str, ids: List[str], include: List[str]
    ) -> Optional[Dict]:
        """Helper to call the vector DB get service (simulated)."""
        if not self.vector_db_mcp_client:
            self.logger.error(
                "Cannot get from vector DB: VectorDBMCP client/config not available."
            )
            return None

        # --- Simulation for local testing ---
        try:
            if hasattr(self.vector_db_mcp_client, "get_embedding"):
                result = self.vector_db_mcp_client.get_embedding(
                    collection_name=collection_name, ids=ids, include=include
                )
                if result and not result.get("error"):
                    return result.get("results")
                else:
                    self.logger.error(
                        f"Simulated vector DB get failed: {result.get('error')}"
                    )
                    return None
            else:
                self.logger.error(
                    "Simulated vector DB client does not have 'get_embedding' method."
                )
                return None
        except Exception as e:
            self.logger.error(f"Error during simulated vector DB get: {e}")
            return None
        # --- End Simulation ---

    async def retrieve_by_text(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        top_k: int = None,
        final_top_k: int = None,
        where_filter: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
        include_metadata: bool = True,
        include_text: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve relevant documents based on a text query."""
        start_time = time.time()
        name = collection_name or self.default_collection_name
        initial_top_k = top_k or (self.default_top_k * 2)  # Retrieve more if reranking
        final_top_k = final_top_k or self.default_top_k

        if not query_text or not isinstance(query_text, str):
            return {
                "results": [],
                "error": "query_text must be a non-empty string.",
                "processing_time": time.time() - start_time,
            }

        if not self.embeddings_service_available:
            return {
                "results": [],
                "error": "Embeddings service is unavailable.",
                "processing_time": time.time() - start_time,
            }
        if not self.vector_db_service_available:
            return {
                "results": [],
                "error": "Vector DB service is unavailable.",
                "processing_time": time.time() - start_time,
            }

        try:
            # 1. Generate query embedding
            query_vector = await self._call_embedding_service(query_text)
            if query_vector is None:
                raise ValueError("Failed to generate query embedding.")

            # 2. Query Vector DB
            include_options = ["distances"]
            if include_metadata:
                include_options.append("metadatas")
            # Include documents if needed for reranking or final output
            if (use_reranker and self.reranker) or include_text:
                include_options.append("documents")

            query_results = await self._call_vector_db_query(
                collection_name=name,
                query_vector=[query_vector],  # Pass as list of vectors
                top_k=initial_top_k,
                where_filter=where_filter,
                include=include_options,
            )

            if query_results is None:
                raise ValueError("Failed to query vector database.")

            #
            # Check if results are structured as expected (e.g., Chroma's
            # format)
            #
            # Assuming query_results is like {'ids': [[]], 'distances': [[]],
            # 'metadatas': [[]], 'documents': [[]]}
            if not query_results or not all(
                k in query_results for k in ["ids", "distances"]
            ):
                self.logger.warning(
                    f"Unexpected format from vector DB query: {query_results}"
                )
                return {
                    "results": [],
                    "collection_name": name,
                    "processing_time": time.time() - start_time,
                }

            # Extract results for the single query
            ids = query_results["ids"][0] if query_results.get("ids") else []
            distances = (
                query_results["distances"][0] if query_results.get("distances") else []
            )
            metadatas = (
                query_results["metadatas"][0]
                if query_results.get("metadatas")
                else [None] * len(ids)
            )
            documents = (
                query_results["documents"][0]
                if query_results.get("documents")
                else [None] * len(ids)
            )

            if not ids:
                return {
                    "results": [],
                    "collection_name": name,
                    "processing_time": time.time() - start_time,
                }

            # Combine initial results
            initial_results = []
            for i in range(len(ids)):
                initial_results.append(
                    {
                        "id": ids[i],
                        "distance": distances[i],
                        "score": 1.0 - distances[i]
                        if distances[i] is not None
                        else 0.0,  # Convert distance to similarity score
                        "metadata": metadatas[i],
                        "text_snippet": documents[
                            i
                        ],  # Assuming 'documents' field contains the text
                    }
                )

            # 3. Rerank if requested and possible
            final_results = initial_results
            if (
                use_reranker
                and self.reranker
                and HAVE_CROSS_ENCODER
                and documents[0] is not None
            ):
                self.logger.info(
                    f"Reranking top {len(initial_results)} results using {self.reranker_model_name}..."
                )
                pairs = [
                    [query_text, res["text_snippet"]]
                    for res in initial_results
                    if res["text_snippet"]
                ]

                if pairs:
                    # Compute cross-encoder scores
                    cross_scores = self.reranker.predict(pairs)

                    # Add scores to results and sort
                    for i in range(len(initial_results)):
                        if initial_results[i]["text_snippet"]:
                            #
                            # Find corresponding score (handle cases where
                            # some snippets were None)
                            pair_index = next(
                                (
                                    idx
                                    for idx, p in enumerate(pairs)
                                    if p[1] == initial_results[i]["text_snippet"]
                                ),
                                -1,
                            )
                            if pair_index != -1:
                                initial_results[i]["rerank_score"] = float(
                                    cross_scores[pair_index]
                                )
                        else:
                            initial_results[i]["rerank_score"] = -float(
                                "inf"
                            )  # Penalize missing text

                    # Sort by rerank score (higher is better)
                    final_results = sorted(
                        initial_results,
                        key=lambda x: x.get("rerank_score", -float("inf")),
                        reverse=True,
                    )
                else:
                    self.logger.warning(
                        "Could not rerank - no text snippets found in initial results."
                    )
                    # Keep initial results sorted by vector similarity
                    final_results = sorted(
                        initial_results, key=lambda x: x.get("score", 0.0), reverse=True
                    )

            else:
                # Sort by initial vector similarity score if not reranking
                final_results = sorted(
                    initial_results, key=lambda x: x.get("score", 0.0), reverse=True
                )

            # 4. Select top K results and format output
            top_final_results = final_results[:final_top_k]

            # Filter output based on include flags
            output_results = []
            for res in top_final_results:
                item = {
                    "id": res["id"],
                    "score": res.get("rerank_score", res.get("score")),
                }  # Use rerank score if available
                if include_metadata:
                    item["metadata"] = res.get("metadata")
                if include_text:
                    item["text_snippet"] = res.get("text_snippet")
                output_results.append(item)

            return {
                "results": output_results,
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error retrieving documents by text: {e}")
            return {
                "results": [],
                "error": str(e),
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

    async def retrieve_by_vector(
        self,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        top_k: int = None,
        where_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_text: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve relevant documents based on a query vector."""
        start_time = time.time()
        name = collection_name or self.default_collection_name
        top_k = top_k or self.default_top_k

        if not query_vector or not isinstance(query_vector, list):
            return {
                "results": [],
                "error": "query_vector must be a non-empty list of numbers.",
                "processing_time": time.time() - start_time,
            }

        if not self.vector_db_service_available:
            return {
                "results": [],
                "error": "Vector DB service is unavailable.",
                "processing_time": time.time() - start_time,
            }

        try:
            # Query Vector DB
            include_options = ["distances"]
            if include_metadata:
                include_options.append("metadatas")
            if include_text:
                include_options.append("documents")

            query_results = await self._call_vector_db_query(
                collection_name=name,
                query_vector=[query_vector],  # Pass as list of vectors
                top_k=top_k,
                where_filter=where_filter,
                include=include_options,
            )

            if query_results is None:
                raise ValueError("Failed to query vector database.")

            #
            # Assuming query_results is like {'ids': [[]], 'distances': [[]],
            # 'metadatas': [[]], 'documents': [[]]}
            if not query_results or not all(
                k in query_results for k in ["ids", "distances"]
            ):
                self.logger.warning(
                    f"Unexpected format from vector DB query: {query_results}"
                )
                return {
                    "results": [],
                    "collection_name": name,
                    "processing_time": time.time() - start_time,
                }

            # Extract results for the single query
            ids = query_results["ids"][0] if query_results.get("ids") else []
            distances = (
                query_results["distances"][0] if query_results.get("distances") else []
            )
            metadatas = (
                query_results["metadatas"][0]
                if query_results.get("metadatas")
                else [None] * len(ids)
            )
            documents = (
                query_results["documents"][0]
                if query_results.get("documents")
                else [None] * len(ids)
            )

            if not ids:
                return {
                    "results": [],
                    "collection_name": name,
                    "processing_time": time.time() - start_time,
                }

            # Format output results
            output_results = []
            for i in range(len(ids)):
                item = {
                    "id": ids[i],
                    "score": 1.0 - distances[i]
                    if distances[i] is not None
                    else 0.0,  # Convert distance to similarity
                }
                if include_metadata:
                    item["metadata"] = metadatas[i]
                if include_text:
                    item["text_snippet"] = documents[i]
                output_results.append(item)

            # Sort by score (higher is better)
            output_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            return {
                "results": output_results,
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error retrieving documents by vector: {e}")
            return {
                "results": [],
                "error": str(e),
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }


# Note: The if __name__ == "__main__": block has been removed as requested.
