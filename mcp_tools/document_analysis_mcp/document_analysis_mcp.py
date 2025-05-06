"""
Document Analysis MCP Tool

This module implements a consolidated Model Context Protocol (MCP) server for
processing, understanding, and retrieving information from financial documents,
leveraging models like LayoutLM for structure and BERT-Fin for embeddings.

This production-ready implementation includes:
- Robust error handling and fallbacks
- Performance optimizations with caching
- Comprehensive logging and metrics
- Efficient resource management
"""

import os
import json

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
import time
import numpy as np
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer

# Import libraries for document processing and embeddings
HAVE_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering, AutoTokenizer, AutoModel
    # LayoutLMv3Processor includes image processing capabilities
    HAVE_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    try:
        # Check if packages are installed but import failed
        import pkg_resources
        import sys
        transformers_installed = False
        torch_installed = False
        
        try:
            pkg_resources.get_distribution('transformers')
            transformers_installed = True
        except pkg_resources.DistributionNotFound:
            pass
            
        try:
            pkg_resources.get_distribution('torch')
            torch_installed = True
        except pkg_resources.DistributionNotFound:
            pass
            
        if transformers_installed or torch_installed:
            print(f"Transformers/torch packages found but import failed. Python path: {sys.path}")
    except Exception:
        pass
    print("Warning: Transformers not installed or import failed. Document analysis features will be limited.")

# PDF processing library
HAVE_PDF_READER = False
try:
    # Using PyMuPDF as an example
    import fitz  # PyMuPDF
    HAVE_PDF_READER = True
except (ImportError, ModuleNotFoundError):
    try:
        # Check if package is installed but import failed
        import pkg_resources
        try:
            pkg_resources.get_distribution('PyMuPDF')
            import sys
            print(f"PyMuPDF package found but import failed. Python path: {sys.path}")
        except pkg_resources.DistributionNotFound:
            pass
    except Exception:
        pass
    print("Warning: PyMuPDF (fitz) not installed or import failed. PDF processing will be limited.")


class DocumentAnalysisMCP(BaseMCPServer):
    """
    Consolidated MCP server for financial document analysis.
    Handles document parsing, layout understanding, embedding generation,
    and information retrieval.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Document Analysis MCP server.

        Args:
            config: Optional configuration dictionary. If None, loads from
                  config/document_analysis_mcp/document_analysis_mcp_config.json
        """
        if config is None:
            config_path = os.path.join("config", "document_analysis_mcp", "document_analysis_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    # Initialize logger first for proper error handling
                    self.logger = NetdataLogger(component_name="document-analysis-mcp")
                    self.logger.error(f"Error loading config from {config_path}: {e}")
                    config = {}
            else:
                # Initialize logger first for proper error handling
                self.logger = NetdataLogger(component_name="document-analysis-mcp")
                self.logger.warning(f"Config file not found at {config_path}. Using default settings.")
                config = {}
        
        # Initialize monitoring
        if not hasattr(self, 'logger'):
            self.logger = NetdataLogger(component_name="document-analysis-mcp")
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.metrics_collector.start()

        super().__init__(name="document_analysis_mcp", config=config)
        
        # Initialize cache and locks
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._last_activity = time.time()
        
        # Set up health monitoring
        self._health_status = {"status": "healthy", "last_updated": datetime.now().isoformat()}
        self._last_health_check = time.time()

        self._configure_from_config()
        self._initialize_models() # Renamed from _initialize_models_and_db
        self._register_tools()
        self._register_management_tools()

        self.logger.info("Document Analysis MCP initialized successfully")

    def _configure_from_config(self):
        """Extract configuration values."""
        # Document Layout Model settings
        layout_config = self.config.get("layout_model", {})
        self.layout_model_path = layout_config.get("model_path", "microsoft/layoutlmv3-base") # Example LayoutLMv3
        self.layout_processor_path = layout_config.get("processor_path", self.layout_model_path)

        # Embeddings Model settings
        embed_config = self.config.get("embeddings_model", {})
        self.embeddings_model_path = embed_config.get("model_path", "sentence-transformers/all-MiniLM-L6-v2") # General purpose default
        self.embedding_dimension = embed_config.get("embedding_dimension", 384) # Dimension for MiniLM

        # Removed Vector DB settings

        # Performance settings
        perf_config = self.config.get("performance", {})
        self.use_gpu = perf_config.get("use_gpu", torch.cuda.is_available() if HAVE_TRANSFORMERS else False)
        self.doc_chunk_size = perf_config.get("doc_chunk_size", 500) # Chars per chunk for embedding
        self.doc_chunk_overlap = perf_config.get("doc_chunk_overlap", 50)
        self.batch_size = perf_config.get("batch_size", 32) # Added batch_size config
        
        # Cache settings
        cache_config = self.config.get("cache", {})
        self.enable_cache = cache_config.get("enable", True)
        self.cache_ttl = cache_config.get("ttl_seconds", 3600)  # Default 1 hour cache TTL
        self.max_cache_size = cache_config.get("max_size", 1000)  # Maximum number of items in cache
        
        # Model settings
        model_config = self.config.get("model_settings", {})
        self.model_load_retries = model_config.get("load_retries", 2)
        self.model_unload_threshold = model_config.get("unload_threshold", 3600)  # Unload after 1 hour of inactivity

        self.logger.info("Document Analysis MCP configuration loaded",
                       layout_model=self.layout_model_path,
                       embeddings_model=self.embeddings_model_path,
                       enable_cache=self.enable_cache,
                       use_gpu=self.use_gpu)
                       
    def _register_management_tools(self):
        """Register tools for monitoring and management of the MCP server."""
        self.register_tool(
            self.get_health_status,
            "get_health_status",
            "Get the current health status of the MCP server"
        )
        self.register_tool(
            self.get_cache_stats,
            "get_cache_stats",
            "Get statistics about the cache usage"
        )
        self.register_tool(
            self.clear_cache,
            "clear_cache",
            "Clear the cache to free up memory"
        )
        self.register_tool(
            self.reload_models,
            "reload_models",
            "Reload all models (useful if models failed to load initially)"
        )
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the MCP server."""
        # Update health status if it's been a while
        if time.time() - self._last_health_check > 60:  # Update every minute
            self._update_health_status()
            
        return {
            "status": self._health_status["status"],
            "models_loaded": self._health_status.get("models_loaded", {}),
            "uptime_seconds": time.time() - self._last_activity,
            "cache_size": len(self._cache),
            "last_updated": self._health_status.get("last_updated", datetime.now().isoformat())
        }
        
    def _update_health_status(self) -> None:
        """Update the health status of the MCP server."""
        self._last_health_check = time.time()
        
        # Check if models are loaded
        models_loaded = {
            "layout": self.layout_model is not None and self.layout_processor is not None,
            "embeddings": self.embeddings_model is not None and self.embeddings_tokenizer is not None
        }
        
        # Determine overall status
        if any(models_loaded.values()):
            status = "healthy"
        else:
            status = "degraded"
            
        self._health_status = {
            "status": status,
            "models_loaded": models_loaded,
            "last_updated": datetime.now().isoformat()
        }
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage."""
        with self._cache_lock:
            current_time = time.time()
            cache_size = len(self._cache)
            
            # Count items by type and age
            method_counts = {}
            age_buckets = {"<1min": 0, "1-10min": 0, "10-60min": 0, ">60min": 0}
            
            for key, (timestamp, _) in self._cache.items():
                # Extract method name from key (first part before the hash)
                method = key.split(":")[0] if ":" in key else "unknown"
                method_counts[method] = method_counts.get(method, 0) + 1
                
                # Categorize by age
                age_seconds = current_time - timestamp
                if age_seconds < 60:
                    age_buckets["<1min"] += 1
                elif age_seconds < 600:
                    age_buckets["1-10min"] += 1
                elif age_seconds < 3600:
                    age_buckets["10-60min"] += 1
                else:
                    age_buckets[">60min"] += 1
            
            return {
                "cache_size": cache_size,
                "cache_enabled": self.enable_cache,
                "cache_ttl_seconds": self.cache_ttl,
                "method_counts": method_counts,
                "age_distribution": age_buckets
            }
    
    def clear_cache(self, method_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the cache to free up memory.
        
        Args:
            method_name: Optional method name to clear cache for. If None, clears all cache.
            
        Returns:
            Dict with information about the operation
        """
        with self._cache_lock:
            if method_name:
                # Count items to be removed
                count_before = len(self._cache)
                # Remove items for the specified method
                keys_to_remove = [
                    key for key in list(self._cache.keys())
                    if key.startswith(method_name)
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                count_after = len(self._cache)
                removed = count_before - count_after
                
                self.logger.info(f"Cleared {removed} items from cache for method {method_name}")
                return {
                    "success": True,
                    "items_removed": removed,
                    "remaining_items": count_after,
                    "method": method_name
                }
            else:
                # Clear all cache
                count = len(self._cache)
                self._cache.clear()
                
                self.logger.info(f"Cleared all {count} items from cache")
                return {
                    "success": True,
                    "items_removed": count,
                    "remaining_items": 0
                }
    
    def reload_models(self) -> Dict[str, Any]:
        """
        Reload all models (useful if models failed to load initially).
        
        Returns:
            Dict with information about the operation
        """
        start_time = time.time()
        self.logger.info("Reloading all models")
        
        # Unload existing models first
        with self._model_lock:
            self.layout_processor = None
            self.layout_model = None
            self.embeddings_model = None
            self.embeddings_tokenizer = None
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            if self.use_gpu and HAVE_TRANSFORMERS and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Load models again
        self._initialize_models()
        
        # Return status
        processing_time = time.time() - start_time
        return {
            "success": self._models_loaded,
            "models_loaded": self._health_status.get("models_loaded", {}),
            "status": self._health_status["status"],
            "processing_time": processing_time
        }
        
    def _get_cache_key(self, method_name: str, **kwargs) -> str:
        """Generate a cache key for the given method and arguments."""
        # Create a string representation of the kwargs, sorted by key
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        # Create a hash of the method name and kwargs
        key = hashlib.md5(f"{method_name}:{kwargs_str}".encode()).hexdigest()
        return f"{method_name}:{key}"

    def _get_from_cache(self, method_name: str, **kwargs) -> Optional[Any]:
        """Get a cached result if available and not expired."""
        if not self.enable_cache:
            return None
            
        with self._cache_lock:
            cache_key = self._get_cache_key(method_name, **kwargs)
            cached_item = self._cache.get(cache_key)
            
            if cached_item is None:
                return None
                
            timestamp, result = cached_item
            if time.time() - timestamp > self.cache_ttl:
                # Cache entry has expired
                del self._cache[cache_key]
                return None
                
            self.logger.debug(f"Cache hit for {method_name}", cache_key=cache_key)
            return result

    def _add_to_cache(self, method_name: str, result: Any, **kwargs) -> None:
        """Add a result to the cache."""
        if not self.enable_cache:
            return
            
        with self._cache_lock:
            cache_key = self._get_cache_key(method_name, **kwargs)
            self._cache[cache_key] = (time.time(), result)
            self.logger.debug(f"Added to cache: {method_name}", cache_key=cache_key)
            
            # Clean up old cache entries if cache is getting too large
            if len(self._cache) > self.max_cache_size:
                self._clean_cache()

    def _clean_cache(self) -> None:
        """Remove expired items from the cache."""
        with self._cache_lock:
            current_time = time.time()
            expired_keys = [
                key for key, (timestamp, _) in self._cache.items()
                if current_time - timestamp > self.cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            self.logger.info(f"Cleaned {len(expired_keys)} expired items from cache")
            
    def shutdown(self) -> None:
        """
        Clean up resources when shutting down the MCP server.
        This method ensures proper cleanup of models and other resources.
        """
        self.logger.info("Shutting down DocumentAnalysisMCP server")
        
        # Clean up cache
        with self._cache_lock:
            cache_size = len(self._cache)
            self._cache.clear()
            self.logger.info(f"Cleared {cache_size} items from cache")
        
        # Clean up models
        with self._model_lock:
            # Free GPU memory if using CUDA
            if self.use_gpu and HAVE_TRANSFORMERS and torch.cuda.is_available():
                if self.embeddings_model is not None:
                    self.embeddings_model.cpu()
                if self.layout_model is not None:
                    self.layout_model.cpu()
                torch.cuda.empty_cache()
                self.logger.info("Cleared GPU memory")
            
            # Set models to None to help garbage collection
            self.layout_processor = None
            self.layout_model = None
            self.embeddings_model = None
            self.embeddings_tokenizer = None
            
            self.logger.info("Models unloaded")
        
        # Stop metrics collector
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            self.metrics_collector.stop()
            self.logger.info("Metrics collector stopped")
        
        # Call parent shutdown method
        super().shutdown()

    def _initialize_models(self): # Renamed from _initialize_models_and_db
        """Initialize models."""
        self.layout_processor = None
        self.layout_model = None
        self.embeddings_model = None
        self.embeddings_tokenizer = None
        # Removed vector DB client/collection initialization
        self._models_loaded = False

        if not HAVE_TRANSFORMERS:
            self.logger.warning("Transformers not available. Document models cannot be loaded.")
            return

        device = torch.device("cuda" if self.use_gpu else "cpu")

        try:
            # Load Layout Model and Processor
            self.logger.info(f"Loading Layout model: {self.layout_model_path}")
            start_time = time.time()
            self.layout_processor = AutoProcessor.from_pretrained(self.layout_processor_path)
            self.layout_model = AutoModelForDocumentQuestionAnswering.from_pretrained(self.layout_model_path)
            self.layout_model.to(device)
            load_time = time.time() - start_time
            self.logger.timing("model_load_time_ms.layout", load_time * 1000)
            # Separate the logging from timing - use keyword args with info() not timing()
            self.logger.info(f"Layout model loaded in {load_time:.2f}s",
                           model_name=self.layout_model_path,
                           load_time_seconds=load_time)

            # Load Embeddings Model
            self.logger.info(f"Loading Embeddings model: {self.embeddings_model_path}")
            start_time = time.time()
            self.embeddings_tokenizer = AutoTokenizer.from_pretrained(self.embeddings_model_path)
            self.embeddings_model = AutoModel.from_pretrained(self.embeddings_model_path)
            self.embeddings_model.to(device)
            load_time = time.time() - start_time
            self.logger.timing("model_load_time_ms.embeddings", load_time * 1000)
            # Separate the logging from timing - use keyword args with info() not timing()
            self.logger.info(f"Embeddings model loaded in {load_time:.2f}s",
                           model_name=self.embeddings_model_path,
                           load_time_seconds=load_time)

            self._models_loaded = True

        except Exception as e:
            self.logger.error(f"Error loading document/embedding models: {e}", exc_info=True)
            self._models_loaded = False

        # Vector DB initialization removed

    def _register_tools(self):
        """Register document analysis tools."""
        self.register_tool(
            self.process_document,
            "process_document",
            "Process a document (e.g., PDF) to extract text and structure."
        )
        self.register_tool(
            self.extract_structured_data,
            "extract_structured_data",
            "Extract structured data (e.g., tables, key-value pairs) from a document image or PDF page."
        )
        self.register_tool(
            self.answer_question_from_document,
            "answer_question_from_document",
            "Answer a question based on the content of a document image or PDF page."
        )
        # Removed vector store related tools


    def _extract_text_from_pdf(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Extract text and basic layout info from each page of a PDF."""
        if not HAVE_PDF_READER:
            self.logger.error("PyMuPDF (fitz) required for PDF processing.")
            return None
        if not os.path.exists(file_path):
            self.logger.error(f"PDF file not found: {file_path}")
            return None

        pages_content = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                pages_content.append({"page": page_num + 1, "text": text})
            doc.close()
            return pages_content
        except Exception as e:
            self.logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        # This might still be useful even without vector DB for other processing
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.doc_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.doc_chunk_size - self.doc_chunk_overlap
            if start >= len(text): break
            start = min(start, len(text) - self.doc_chunk_overlap // 2) if len(text) > self.doc_chunk_overlap else start
        if not chunks: return [text] if text else []
        return chunks

    def _generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts using the loaded model."""
        # This might still be useful for other purposes, keeping it for now
        if not self._models_loaded or self.embeddings_model is None or self.embeddings_tokenizer is None:
            self.logger.error("Embeddings model not loaded for batch generation.")
            return None
        try:
            device = torch.device("cuda" if self.use_gpu else "cpu")
            inputs = self.embeddings_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = self.embeddings_model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            return embeddings.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error generating embeddings batch: {e}", exc_info=True)
            return None

    # --- Tool Implementations ---

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document (PDF) to extract text."""
        # Update last activity timestamp
        self._last_activity = time.time()
        
        # Check cache first
        cache_key_params = {"file_path": file_path}
        cached_result = self._get_from_cache("process_document", **cache_key_params)
        if cached_result is not None:
            self.logger.info("Using cached document processing result")
            cached_result["from_cache"] = True
            return cached_result
            
        start_time = time.time()
        self.logger.info(f"Processing document: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            return {"error": "Currently only PDF documents are supported."}

        pages_content = self._extract_text_from_pdf(file_path)
        if pages_content is None:
            return {"error": f"Failed to extract text from PDF: {file_path}"}

        doc_id = os.path.basename(file_path)
        # Removed vector store processing logic
        total_pages = len(pages_content)

        processing_time = time.time() - start_time
        self.logger.timing(f"process_document_time_ms.pages_{total_pages}", processing_time * 1000)

        result = {
            "status": "success",
            "doc_id": doc_id,
            "page_count": total_pages,
            "processing_time": processing_time,
            "extracted_text": pages_content # Return the extracted text per page
        }
        
        # Cache the result
        self._add_to_cache("process_document", result, **cache_key_params)
        
        return result

    def extract_structured_data(self, file_path: str, page_num: int = 1) -> Dict[str, Any]:
        """Extract structured data (tables, key-value) from a specific page (image/PDF)."""
        self.logger.warning("Structured data extraction requires a specialized model which is not fully implemented here.")
        return {"error": "Structured data extraction not fully implemented."}

    def answer_question_from_document(self, file_path: str, question: str, page_num: int = 1) -> Dict[str, Any]:
        """Answer a question based on a document page (image/PDF)."""
        start_time = time.time()
        if not self._models_loaded or self.layout_model is None or self.layout_processor is None:
            return {"error": "Document Q&A model not loaded."}
        if not HAVE_PDF_READER:
             return {"error": "PDF reader (PyMuPDF) not installed."}

        self.logger.info(f"Answering question on document: {file_path} (page {page_num})", question=question)

        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                 doc.close()
                 return {"error": f"Invalid page number {page_num}. Document has {len(doc)} pages."}
            page = doc.load_page(page_num - 1)
            try:
                 from PIL import Image
                 pix = page.get_pixmap()
                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            except ImportError:
                 doc.close()
                 return {"error": "Pillow library required for image conversion."}
            doc.close()

            device = torch.device("cuda" if self.use_gpu else "cpu")
            encoding = self.layout_processor(img, question, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = self.layout_model(**encoding)

            start_index = torch.argmax(outputs.start_logits)
            end_index = torch.argmax(outputs.end_logits)
            answer_tokens = encoding.input_ids[0][start_index : end_index + 1]
            answer = self.layout_processor.tokenizer.decode(answer_tokens)

            processing_time = time.time() - start_time
            self.logger.timing("doc_qa_time_ms", processing_time * 1000)

            return {
                "answer": answer.strip(),
                "question": question,
                "page": page_num,
                "confidence_start": round(outputs.start_logits.max().item(), 4),
                "confidence_end": round(outputs.end_logits.max().item(), 4),
                "processing_time": processing_time
            }

        except Exception as e:
            self.logger.error(f"Error answering question from document: {e}", exc_info=True)
            return {"error": str(e)}
