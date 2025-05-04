"""
Vector Database MCP Tool

This module implements a Model Context Protocol (MCP) server for interacting
with a vector database (e.g., Chroma, FAISS, Pinecone) to store and query
vector embeddings.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import vector database libraries - use placeholders for flexibility
try:
    import chromadb

    HAVE_CHROMA = True
except ImportError:
    HAVE_CHROMA = False
    chromadb = None

try:
    import faiss

    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    faiss = None

# Add imports for other vector DBs like Pinecone
try:
    import pinecone

    HAVE_PINECONE = True
except ImportError:
    HAVE_PINECONE = False
    pinecone = None

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-vector-db",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class VectorDBMCP(BaseMCPServer):
    """
    MCP server for managing vector embeddings in a vector database.

    Provides tools to store, query, update, and delete vector embeddings
    and associated metadata, enabling semantic search and similarity tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Vector Database MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - db_type: Type of vector database ('chroma', 'faiss', 'pinecone', etc.)
                - db_path: Path for persistent storage (for Chroma/FAISS)
                - host: Host address for remote DBs
                - api_key: API key for cloud-based DBs (e.g., Pinecone)
                - default_collection: Default collection name to use
                - embedding_dimension: Expected dimension of embeddings (required for FAISS)
        """
        super().__init__(name="vector_db_mcp", config=config)

        # Set configurations with defaults
        self.db_type = self.config.get("db_type", "chroma")  # Default to Chroma
        self.db_path = self.config.get("db_path", "./vector_db_storage")
        self.host = self.config.get("host")
        self.api_key = self.config.get("api_key")
        self.default_collection_name = self.config.get(
            "default_collection", "financial_context"
        )
        self.embedding_dimension = self.config.get(
            "embedding_dimension"
        )  # Required for FAISS

        # Validate configuration based on db_type
        if self.db_type == "faiss" and not self.embedding_dimension:
            raise ValueError("embedding_dimension is required in config for FAISS.")
        if self.db_type == "pinecone" and (not self.host or not self.api_key):
            raise ValueError("host and api_key are required in config for Pinecone.")

        # Initialize the vector database client/index
        self.client = None
        self.collections: Dict[str, Any] = {}  # Store collection/index objects
        self._init_db()

        # Register tools
        self._register_tools()

    def _init_db(self):
        """Initialize connection to the specified vector database."""
        try:
            self.logger.info(
                f"Initializing vector database connection (type: {self.db_type})"
            )

            if self.db_type == "chroma":
                if not HAVE_CHROMA:
                    raise ImportError(
                        "ChromaDB library not found. Install with: pip install chromadb"
                    )

                # Initialize Chroma client (persistent or in-memory)
                if self.db_path:
                    self.client = chromadb.PersistentClient(path=self.db_path)
                    self.logger.info(
                        f"Initialized persistent ChromaDB client at {self.db_path}"
                    )
                else:
                    self.client = chromadb.Client()
                    self.logger.info("Initialized in-memory ChromaDB client.")
                # Pre-load default collection if it exists
                self._get_collection(self.default_collection_name)

            elif self.db_type == "faiss":
                if not HAVE_FAISS:
                    raise ImportError(
                        "FAISS library not found. Install with: pip install faiss-cpu or faiss-gpu"
                    )

                #
                # FAISS requires manual index management. We'll use dicts to
                # store indices and metadata.
                self.faiss_indices: Dict[str, Any] = {}  # type: ignore  # faiss.Index may not be available
                self.faiss_metadata: Dict[
                    str, Dict[str, Any]
                ] = {}  # Store metadata associated with IDs
                self.faiss_next_id: Dict[
                    str, int
                ] = {}  # Track next available ID for each index
                self.logger.info("Initialized FAISS in-memory storage.")
                #
                # Note: Persistence for FAISS needs separate handling (e.g.,
                # saving index to disk)

            elif self.db_type == "pinecone":
                if not HAVE_PINECONE:
                    raise ImportError(
                        "Pinecone library not found. Install with: pip install pinecone-client"
                    )

                if not self.api_key:
                    raise ValueError("api_key is required in config for Pinecone.")

                # Initialize Pinecone client
                self.client = pinecone.Pinecone(api_key=self.api_key)

                # Get or create the index
                self.index_name = self.config.get(
                    "pinecone_index_name", self.default_collection_name
                )

                # Check if index exists
                indexes = self.client.list_indexes()
                if (
                    not hasattr(indexes, "names")
                    or self.index_name not in indexes.names()
                ):
                    # Create the index if it doesn't exist
                    if not self.embedding_dimension:
                        raise ValueError(
                            "embedding_dimension is required to create a new Pinecone index"
                        )

                    # Create index configuration
                    metric = self.config.get("pinecone_metric", "cosine")
                    server_spec = pinecone.ServerlessSpec(
                        cloud=self.config.get("pinecone_cloud", "aws"),
                        region=self.config.get("pinecone_region", "us-west-2"),
                    )

                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.embedding_dimension,
                        metric=metric,
                        spec=server_spec,
                    )
                    self.logger.info(f"Created new Pinecone index: {self.index_name}")

                # Connect to the index
                self.collections[self.index_name] = self.client.Index(self.index_name)
                self.logger.info(f"Connected to Pinecone index: {self.index_name}")

            else:
                raise ValueError(f"Unsupported vector database type: {self.db_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            self.client = None  # Ensure client is None if init fails

    def _get_collection(self, collection_name: Optional[str] = None) -> Any:
        """Get or create a collection/index object."""
        name = collection_name or self.default_collection_name

        if name in self.collections:
            return self.collections[name]

        if self.client is None:
            raise ConnectionError("Vector database client is not initialized.")

        try:
            if self.db_type == "chroma":
                if isinstance(self.client, chromadb.Client):
                    collection = self.client.get_or_create_collection(name)
                    self.collections[name] = collection
                    self.logger.info(f"Accessed/Created Chroma collection: {name}")
                    return collection
                else:
                    raise TypeError("Chroma client is not of expected type.")

            elif self.db_type == "faiss":
                if name not in self.faiss_indices:
                    # Create a new FAISS index (e.g., IndexFlatL2)
                    index = faiss.IndexFlatL2(self.embedding_dimension)
                    #
                    # Wrap with IndexIDMap for string IDs (optional, requires
                    # managing mapping)
                    # Or use sequential integer IDs
                    self.faiss_indices[name] = index
                    self.faiss_metadata[name] = {}
                    self.faiss_next_id[name] = 0
                    self.logger.info(
                        f"Created new FAISS index: {name} with dimension {self.embedding_dimension}"
                    )
                return self.faiss_indices[name]

            elif self.db_type == "pinecone":
                if name not in self.collections:
                    if not self.client:
                        raise ConnectionError("Pinecone client is not initialized.")

                    # Check if the index exists
                    indexes = self.client.list_indexes()
                    if not hasattr(indexes, "names") or name not in indexes.names():
                        # Create new index if it doesn't exist
                        if not self.embedding_dimension:
                            raise ValueError(
                                "embedding_dimension is required to create a new Pinecone index"
                            )

                        # Create index configuration
                        metric = self.config.get("pinecone_metric", "cosine")
                        server_spec = pinecone.ServerlessSpec(
                            cloud=self.config.get("pinecone_cloud", "aws"),
                            region=self.config.get("pinecone_region", "us-west-2"),
                        )

                        self.client.create_index(
                            name=name,
                            dimension=self.embedding_dimension,
                            metric=metric,
                            spec=server_spec,
                        )
                        self.logger.info(f"Created new Pinecone index: {name}")

                    # Connect to the index
                    self.collections[name] = self.client.Index(name)
                    self.logger.info(f"Connected to Pinecone index: {name}")

                return self.collections[name]

            else:
                raise ValueError(f"Unsupported operation for db_type: {self.db_type}")

        except Exception as e:
            self.logger.error(f"Error accessing/creating collection '{name}': {e}")
            raise

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "add_embeddings",
            self.add_embeddings,
            "Add vector embeddings and associated metadata to a collection",
            {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection to add to (defaults to default collection)",
                    "required": False,
                },
                "embeddings": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "List of vector embeddings to add",
                },
                "metadata": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of metadata dictionaries corresponding to each embedding",
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of unique IDs for each embedding",
                },
            },
            {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "added_count": {"type": "integer"},
                    "collection_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

        self.register_tool(
            "query_similar",
            self.query_similar,
            "Query for embeddings similar to a given vector or text",
            {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection to query (defaults to default collection)",
                    "required": False,
                },
                "query_embeddings": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "List of query vector embeddings",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of similar results to return for each query",
                    "default": 5,
                },
                "where_filter": {
                    "type": "object",
                    "description": "Filter conditions for metadata (Chroma specific)",
                    "required": False,
                },
                "include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to include in results ('metadata', 'documents', 'distances', 'embeddings')",
                    "default": ["metadata", "distances"],
                },
            },
            {
                "type": "object",
                "properties": {
                    "results": {"type": "object"},  # Structure depends on the vector DB
                    "collection_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

        self.register_tool(
            "get_embedding",
            self.get_embedding,
            "Retrieve an embedding and its metadata by ID",
            {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection (defaults to default collection)",
                    "required": False,
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of IDs to retrieve",
                },
                "include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to include ('metadata', 'documents', 'embeddings')",
                    "default": ["metadata", "embeddings"],
                },
            },
            {
                "type": "object",
                "properties": {
                    "results": {"type": "object"},  # Structure depends on the vector DB
                    "collection_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

        self.register_tool(
            "delete_embeddings",
            self.delete_embeddings,
            "Delete embeddings from a collection by ID",
            {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection (defaults to default collection)",
                    "required": False,
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of IDs to delete",
                },
            },
            {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "deleted_count": {
                        "type": "integer"
                    },  # May not be available for all DBs
                    "collection_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

        self.register_tool(
            "list_collections",
            self.list_collections,
            "List all available collections in the vector database",
            {},
            {
                "type": "object",
                "properties": {
                    "collections": {"type": "array", "items": {"type": "string"}},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: List[str],
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add embeddings, metadata, and IDs to a collection."""
        start_time = time.time()
        name = collection_name or self.default_collection_name

        if not embeddings or not metadata or not ids:
            return {
                "success": False,
                "error": "Embeddings, metadata, and IDs lists must be provided and non-empty.",
                "processing_time": time.time() - start_time,
            }

        if not (len(embeddings) == len(metadata) == len(ids)):
            return {
                "success": False,
                "error": "Embeddings, metadata, and IDs lists must have the same length.",
                "processing_time": time.time() - start_time,
            }

        try:
            collection = self._get_collection(name)

            if self.db_type == "chroma":
                if isinstance(collection, chromadb.Collection):
                    # Chroma expects numpy arrays for embeddings if possible
                    embeddings_np = np.array(embeddings, dtype=np.float32)
                    collection.add(
                        embeddings=embeddings_np, metadatas=metadata, ids=ids
                    )
                    added_count = len(ids)
                else:
                    raise TypeError(
                        "Retrieved Chroma collection is not of expected type."
                    )

            elif self.db_type == "faiss":
                if isinstance(collection, faiss.Index):
                    index = collection
                    embeddings_np = np.array(embeddings, dtype=np.float32)

                    # Check embedding dimension
                    if embeddings_np.shape[1] != self.embedding_dimension:
                        raise ValueError(
                            f"Input embedding dimension {embeddings_np.shape[1]} does not match index dimension {self.embedding_dimension}"
                        )

                    # Generate sequential IDs for FAISS
                    start_id = self.faiss_next_id.get(name, 0)
                    faiss_ids = np.arange(start_id, start_id + len(ids))

                    # Add embeddings to FAISS index
                    index.add_with_ids(embeddings_np, faiss_ids)

                    # Store metadata and original IDs mapping
                    collection_metadata = self.faiss_metadata.setdefault(name, {})
                    for i, original_id in enumerate(ids):
                        faiss_id = faiss_ids[i]
                        collection_metadata[str(faiss_id)] = {
                            "original_id": original_id,
                            **metadata[i],
                        }

                    # Update next ID
                    self.faiss_next_id[name] = start_id + len(ids)
                    added_count = len(ids)
                else:
                    raise TypeError("Retrieved FAISS index is not of expected type.")

            elif self.db_type == "pinecone":
                if hasattr(collection, "upsert"):
                    # Prepare data for Pinecone format
                    vectors = []
                    for i, (id_val, embedding, meta) in enumerate(
                        zip(ids, embeddings, metadata)
                    ):
                        vector_entry = {
                            "id": id_val,
                            "values": embedding,
                            "metadata": meta,
                        }
                        vectors.append(vector_entry)

                    #
                    # Upsert data in batches (Pinecone may have batch size
                    # limits)
                    batch_size = 100  # Adjust based on Pinecone recommendations
                    added_count = 0

                    for i in range(0, len(vectors), batch_size):
                        batch = vectors[i : i + batch_size]
                        collection.upsert(vectors=batch)
                        added_count += len(batch)
                else:
                    raise TypeError("Retrieved Pinecone index is not of expected type.")

            else:
                raise ValueError(f"Unsupported operation for db_type: {self.db_type}")

            return {
                "success": True,
                "added_count": added_count,
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error adding embeddings to collection '{name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

    def query_similar(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        collection_name: Optional[str] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query for similar embeddings."""
        start_time = time.time()
        name = collection_name or self.default_collection_name
        include = include or ["metadata", "distances"]  # Default include

        if not query_embeddings:
            return {
                "results": None,
                "error": "query_embeddings must be provided.",
                "processing_time": time.time() - start_time,
            }

        try:
            collection = self._get_collection(name)
            query_embeddings_np = np.array(query_embeddings, dtype=np.float32)

            results = None

            if self.db_type == "chroma":
                if isinstance(collection, chromadb.Collection):
                    # Ensure include list contains valid options for Chroma
                    valid_include = [
                        item
                        for item in include
                        if item in ["metadatas", "documents", "distances", "embeddings"]
                    ]
                    if "metadata" in include and "metadatas" not in valid_include:
                        valid_include.append("metadatas")  # Map metadata -> metadatas

                    query_results = collection.query(
                        query_embeddings=query_embeddings_np.tolist(),  # Chroma query expects list
                        n_results=n_results,
                        where=where_filter,  # Pass Chroma specific filter
                        include=valid_include,
                    )
                    # Reformat results slightly for consistency if needed
                    results = query_results
                else:
                    raise TypeError(
                        "Retrieved Chroma collection is not of expected type."
                    )

            elif self.db_type == "faiss":
                if isinstance(collection, faiss.Index):
                    index = collection
                    # Check query embedding dimension
                    if query_embeddings_np.shape[1] != self.embedding_dimension:
                        raise ValueError(
                            f"Query embedding dimension {query_embeddings_np.shape[1]} does not match index dimension {self.embedding_dimension}"
                        )

                    # Perform search
                    distances, faiss_ids = index.search(query_embeddings_np, n_results)

                    # Retrieve metadata using the returned FAISS IDs
                    collection_metadata = self.faiss_metadata.get(name, {})
                    query_results = {
                        "ids": [],
                        "distances": [],
                        "metadatas": [],
                        "embeddings": [],  # Embeddings retrieval needs separate step if using IndexIDMap
                        "documents": [],  # Documents not stored directly in FAISS index
                    }

                    for i in range(len(faiss_ids)):  # Iterate through each query result
                        ids_list = []
                        dist_list = []
                        meta_list = []

                        for j in range(
                            len(faiss_ids[i])
                        ):  # Iterate through top k results for query i
                            faiss_id = faiss_ids[i][j]
                            if faiss_id != -1:  # FAISS returns -1 for no result
                                meta = collection_metadata.get(str(faiss_id))
                                if meta:
                                    ids_list.append(
                                        meta.get("original_id", str(faiss_id))
                                    )  # Return original ID
                                    dist_list.append(float(distances[i][j]))
                                    if "metadata" in include:
                                        meta_copy = meta.copy()
                                        meta_copy.pop(
                                            "original_id", None
                                        )  # Don't include internal mapping ID
                                        meta_list.append(meta_copy)
                                    else:
                                        meta_list.append(
                                            None
                                        )  # Add placeholder if metadata not requested
                                else:
                                    #
                                    # Handle case where metadata might be
                                    # missing (should not happen ideally)
                                    ids_list.append(str(faiss_id))
                                    dist_list.append(float(distances[i][j]))
                                    meta_list.append(None)

                        query_results["ids"].append(ids_list)
                        query_results["distances"].append(dist_list)
                        if "metadata" in include:
                            query_results["metadatas"].append(meta_list)

                    # Filter results based on 'include'
                    final_results = {}
                    if "ids" in query_results:
                        final_results["ids"] = query_results["ids"]
                    if "distances" in include and "distances" in query_results:
                        final_results["distances"] = query_results["distances"]
                    if "metadata" in include and "metadatas" in query_results:
                        final_results["metadatas"] = query_results["metadatas"]
                    #
                    # Add documents/embeddings if they were included and
                    # retrieved

                    results = final_results
                else:
                    raise TypeError("Retrieved FAISS index is not of expected type.")

            # Add logic for other DB types here

            else:
                raise ValueError(f"Unsupported operation for db_type: {self.db_type}")

            return {
                "results": results,
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error querying collection '{name}': {e}")
            return {
                "results": None,
                "error": str(e),
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

    def get_embedding(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve embeddings and metadata by ID."""
        start_time = time.time()
        name = collection_name or self.default_collection_name
        include = include or ["metadata", "embeddings"]  # Default include

        if not ids:
            return {
                "results": None,
                "error": "IDs list must be provided and non-empty.",
                "processing_time": time.time() - start_time,
            }

        try:
            collection = self._get_collection(name)
            results = None

            if self.db_type == "chroma":
                if isinstance(collection, chromadb.Collection):
                    # Ensure include list contains valid options for Chroma
                    valid_include = [
                        item
                        for item in include
                        if item in ["metadatas", "documents", "embeddings"]
                    ]
                    if "metadata" in include and "metadatas" not in valid_include:
                        valid_include.append("metadatas")

                    get_results = collection.get(ids=ids, include=valid_include)
                    # Reformat results slightly for consistency if needed
                    results = get_results
                else:
                    raise TypeError(
                        "Retrieved Chroma collection is not of expected type."
                    )

            elif self.db_type == "faiss":
                #
                # FAISS retrieval by original ID requires searching the
                # metadata mapping
                if isinstance(collection, faiss.Index):
                    index = collection
                    collection_metadata = self.faiss_metadata.get(name, {})

                    # Find the FAISS IDs corresponding to the original IDs
                    faiss_id_map = {
                        v.get("original_id"): k for k, v in collection_metadata.items()
                    }
                    target_faiss_ids_str = [
                        faiss_id_map.get(original_id) for original_id in ids
                    ]
                    target_faiss_ids = [
                        int(fid) for fid in target_faiss_ids_str if fid is not None
                    ]

                    if not target_faiss_ids:
                        results = {
                            "ids": [],
                            "embeddings": [],
                            "metadatas": [],
                        }  # Return empty if no IDs found
                    else:
                        #
                        # FAISS reconstruct method retrieves vectors by
                        # index ID
                        retrieved_embeddings = index.reconstruct_batch(
                            np.array(target_faiss_ids, dtype=np.int64)
                        )

                        # Prepare results structure
                        get_results = {"ids": [], "embeddings": [], "metadatas": []}
                        original_id_to_faiss_id = {
                            v: k for k, v in faiss_id_map.items()
                        }

                        for i, faiss_id in enumerate(target_faiss_ids):
                            original_id = original_id_to_faiss_id.get(str(faiss_id))
                            if original_id:
                                get_results["ids"].append(original_id)
                                if "embeddings" in include:
                                    get_results["embeddings"].append(
                                        retrieved_embeddings[i].tolist()
                                    )
                                if "metadata" in include:
                                    meta = collection_metadata.get(str(faiss_id), {})
                                    meta_copy = meta.copy()
                                    meta_copy.pop("original_id", None)
                                    get_results["metadatas"].append(meta_copy)

                        results = get_results
                else:
                    raise TypeError("Retrieved FAISS index is not of expected type.")

            # Add logic for other DB types here

            else:
                raise ValueError(f"Unsupported operation for db_type: {self.db_type}")

            return {
                "results": results,
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error getting embeddings from collection '{name}': {e}")
            return {
                "results": None,
                "error": str(e),
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

    def delete_embeddings(
        self, ids: List[str], collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete embeddings from a collection by ID."""
        start_time = time.time()
        name = collection_name or self.default_collection_name

        if not ids:
            return {
                "success": False,
                "error": "IDs list must be provided and non-empty.",
                "processing_time": time.time() - start_time,
            }

        try:
            collection = self._get_collection(name)
            deleted_count = 0

            if self.db_type == "chroma":
                if isinstance(collection, chromadb.Collection):
                    # Chroma's delete returns the IDs that were deleted
                    deleted_ids_result = collection.delete(ids=ids)
                    #
                    # Note: Chroma's delete might not directly return a
                    # count.
                    #
                    # We assume success if no exception, count might be
                    # len(ids) if all existed.
                    deleted_count = len(ids)  # Approximation
                else:
                    raise TypeError(
                        "Retrieved Chroma collection is not of expected type."
                    )

            elif self.db_type == "faiss":
                if isinstance(collection, faiss.Index):
                    index = collection
                    collection_metadata = self.faiss_metadata.get(name, {})

                    # Find FAISS IDs to remove
                    faiss_id_map = {
                        v.get("original_id"): k for k, v in collection_metadata.items()
                    }
                    faiss_ids_to_remove_str = [
                        faiss_id_map.get(original_id) for original_id in ids
                    ]
                    faiss_ids_to_remove = np.array(
                        [
                            int(fid)
                            for fid in faiss_ids_to_remove_str
                            if fid is not None
                        ],
                        dtype=np.int64,
                    )

                    if len(faiss_ids_to_remove) > 0:
                        # FAISS remove_ids requires an IDSelector
                        selector = faiss.IDSelectorBatch(faiss_ids_to_remove)
                        num_removed = index.remove_ids(selector)
                        deleted_count = num_removed

                        # Remove metadata
                        for faiss_id_str in faiss_ids_to_remove_str:
                            if faiss_id_str in collection_metadata:
                                del collection_metadata[faiss_id_str]
                    else:
                        deleted_count = 0
                else:
                    raise TypeError("Retrieved FAISS index is not of expected type.")

            # Add logic for other DB types here

            else:
                raise ValueError(f"Unsupported operation for db_type: {self.db_type}")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(
                f"Error deleting embeddings from collection '{name}': {e}"
            )
            return {
                "success": False,
                "error": str(e),
                "collection_name": name,
                "processing_time": time.time() - start_time,
            }

    def list_collections(self) -> Dict[str, Any]:
        """List all available collections."""
        start_time = time.time()

        if self.client is None:
            return {
                "collections": [],
                "error": "Vector database client is not initialized.",
                "processing_time": time.time() - start_time,
            }

        try:
            collection_names = []
            if self.db_type == "chroma":
                if isinstance(self.client, chromadb.Client):
                    collections_list = self.client.list_collections()
                    collection_names = [c.name for c in collections_list]
                else:
                    raise TypeError("Chroma client is not of expected type.")

            elif self.db_type == "faiss":
                #
                # FAISS collections are managed in memory in this
                # implementation
                collection_names = list(self.faiss_indices.keys())

            #
            # Add logic for other DB types here (e.g., Pinecone
            # client.list_indexes().names)

            else:
                raise ValueError(f"Unsupported operation for db_type: {self.db_type}")

            return {
                "collections": collection_names,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return {
                "collections": [],
                "error": str(e),
                "processing_time": time.time() - start_time,
            }


# Note: The if __name__ == "__main__": block has been removed as requested.
