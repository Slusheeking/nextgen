"""
Embeddings MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
text embedding generation capabilities, potentially using financial domain-specific models.
It relies on the sentence-transformers library.
"""

import os
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For embedding models
try:
    from sentence_transformers import SentenceTransformer

    HAVE_SBERT = True
except ImportError:
    HAVE_SBERT = False
    SentenceTransformer = None  # Define for type hinting even if not available

# For potential GPU usage check
try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None  # Define for type hinting

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-embeddings",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class EmbeddingsMCP(BaseMCPServer):
    """
    MCP server for generating text embeddings using sentence transformers.

    Converts text into dense vector representations for semantic tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Embeddings MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - model_name: Name of the Sentence Transformer model.
                - use_gpu: Whether to attempt using GPU if available.
                - batch_size: Batch size for encoding.
                - cache_dir: Directory for caching models.
        """
        super().__init__(name="embeddings_mcp", config=config)

        # Validate required libraries
        if not HAVE_SBERT:
            self.logger.error(
                "Sentence Transformers library not found. Please install it: pip install sentence-transformers"
            )
            raise ImportError(
                "Sentence Transformers library is required for EmbeddingsMCP."
            )

        # Set configurations with defaults
        self.model_name = self.config.get("model_name")
        if not self.model_name:
            default_model = "all-MiniLM-L6-v2"
            self.logger.warning(
                f"No model_name specified in config. Using default: {default_model}"
            )
            self.model_name = default_model

        self.use_gpu = self.config.get("use_gpu", True)
        self.batch_size = self.config.get("batch_size", 32)
        self.cache_dir = self.config.get("cache_dir", "./embedding_models_cache")

        # Determine device
        self.device = "cuda" if self.use_gpu and self._is_cuda_available() else "cpu"

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except OSError as e:
                self.logger.error(
                    f"Failed to create cache directory {self.cache_dir}: {e}"
                )
                #
                # Continue without cache if creation fails, model might
                # download elsewhere
                self.cache_dir = None

        # Initialize the embedding model
        self.model: Any = None  # type: ignore  # SentenceTransformer may be None if import fails
        self._init_model()

        # Register tools
        self._register_tools()

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available via PyTorch."""
        if not HAVE_TORCH:
            return False
        try:
            return torch.cuda.is_available()
        except Exception as e:
            self.logger.warning(f"Error checking CUDA availability: {e}")
            return False

    def _init_model(self):
        """Initialize the Sentence Transformer model."""
        try:
            self.logger.info(
                f"Loading Sentence Transformer model: {self.model_name} on device: {self.device}"
            )
            #
            # Load standard Sentence Transformer model (v4.1.0 does not support
            # 'device' argument)
            self.model = SentenceTransformer(
                self.model_name, cache_folder=self.cache_dir if self.cache_dir else None
            )
            # Move model to GPU if requested and available
            if self.device == "cuda":
                try:
                    self.model = self.model.to("cuda")
                except Exception as e:
                    self.logger.warning(f"Could not move model to CUDA: {e}")
            self.logger.info(
                f"Successfully initialized embedding model: {self.model_name}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize embedding model '{self.model_name}': {e}"
            )
            self.model = None  # Ensure model is None if loading failed

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "generate_embedding",
            self.generate_embedding,
            "Generate a vector embedding for a single text input",
            {
                "text": {
                    "type": "string",
                    "description": "Text to generate embedding for",
                },
                "normalize": {
                    "type": "boolean",
                    "description": "Whether to normalize the embedding vector",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "embedding": {"type": "array", "items": {"type": "number"}},
                    "dimension": {"type": "integer"},
                    "model_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "error": {"type": "string", "required": False},
                },
            },
        )

        self.register_tool(
            "batch_generate_embeddings",
            self.batch_generate_embeddings,
            "Generate vector embeddings for multiple text inputs in batch",
            {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to generate embeddings for",
                },
                "normalize": {
                    "type": "boolean",
                    "description": "Whether to normalize the embedding vectors",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "embeddings": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "dimension": {"type": "integer"},
                    "model_name": {"type": "string"},
                    "processing_time": {"type": "number"},
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "required": False,
                    },
                },
            },
        )

    def generate_embedding(self, text: str, normalize: bool = True) -> Dict[str, Any]:
        """
        Generate a vector embedding for a single text input.

        Args:
            text: Text to generate embedding for.
            normalize: Whether to normalize the embedding vector.

        Returns:
            Dictionary containing the embedding vector and metadata.
        """
        start_time = time.time()

        if self.model is None:
            return {
                "embedding": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": "Embedding model is not loaded.",
                "processing_time": time.time() - start_time,
            }

        if not isinstance(text, str) or not text.strip():
            return {
                "embedding": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": "Input text must be a non-empty string.",
                "processing_time": time.time() - start_time,
            }

        try:
            # Generate embedding
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                batch_size=1,  # Single text
            )[0]  # Get the first (and only) embedding

            dimension = embedding.shape[0]

            return {
                "embedding": embedding.tolist(),  # Convert numpy array to list for JSON
                "dimension": dimension,
                "model_name": self.model_name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error generating embedding for text: {e}")
            return {
                "embedding": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": f"Failed to generate embedding: {str(e)}",
                "processing_time": time.time() - start_time,
            }

    def batch_generate_embeddings(
        self, texts: List[str], normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate vector embeddings for multiple text inputs in batch.

        Args:
            texts: List of texts to generate embeddings for.
            normalize: Whether to normalize the embedding vectors.

        Returns:
            Dictionary containing the list of embeddings and metadata.
        """
        start_time = time.time()

        if self.model is None:
            return {
                "embeddings": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": "Embedding model is not loaded.",
                "processing_time": time.time() - start_time,
            }

        if not isinstance(texts, list) or not texts:
            return {
                "embeddings": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": "Input 'texts' must be a non-empty list of strings.",
                "processing_time": time.time() - start_time,
            }

        # Validate input list contains strings
        if not all(isinstance(t, str) for t in texts):
            return {
                "embeddings": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": "All items in the 'texts' list must be strings.",
                "processing_time": time.time() - start_time,
            }

        try:
            # Generate embeddings in batch
            embeddings_np = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                batch_size=self.batch_size,
                show_progress_bar=False,  # Disable progress bar for server use
            )

            dimension = embeddings_np.shape[1] if embeddings_np.ndim == 2 else None
            embeddings_list = (
                embeddings_np.tolist()
            )  # Convert numpy array to list for JSON

            return {
                "embeddings": embeddings_list,
                "dimension": dimension,
                "model_name": self.model_name,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            return {
                "embeddings": None,
                "dimension": None,
                "model_name": self.model_name,
                "error": f"Failed to generate batch embeddings: {str(e)}",
                "processing_time": time.time() - start_time,
            }


# Note: The if __name__ == "__main__": block has been removed as requested
# to ensure the code is production-ready without examples or test runs.
