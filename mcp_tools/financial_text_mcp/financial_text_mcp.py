#!/usr/bin/env python3
"""
Financial Text MCP Tool

This module implements a consolidated Model Context Protocol (MCP) server that provides
comprehensive financial text processing capabilities, including entity extraction,
sentiment analysis, and query processing using state-of-the-art pre-trained models.

This production-ready implementation includes:
- Robust error handling and fallbacks
- Performance optimizations with ONNX and quantization
- Comprehensive logging and metrics
- Efficient caching and resource management
"""

import os
import json
import time
import re
import threading
import importlib
from typing import Dict, List, Any, Optional
from datetime import datetime

# Direct imports instead of dynamic loading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

# For pre-trained models with proper error handling
HAVE_TRANSFORMERS = False
try:
    import torch
    try:
        import transformers
        HAVE_TRANSFORMERS = True
        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
        AutoModelForTokenClassification = transformers.AutoModelForTokenClassification
        AutoModel = transformers.AutoModel
        pipeline = transformers.pipeline
    except ImportError:
        print("Warning: Transformers not installed or import failed. Pre-trained model features will be limited.")
except ImportError:
    print("Warning: PyTorch not installed or import failed. Pre-trained model features will be limited.")

# For ONNX optimization
try:
    import onnxruntime as ort
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False
    print("Warning: ONNX Runtime not installed or import failed. Model optimization will be limited.")

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer

class FinancialTextMCP(BaseMCPServer):
    """
    Consolidated MCP server for financial text processing.

    This tool combines entity extraction, sentiment analysis, and query processing
    capabilities using state-of-the-art pre-trained models specialized for financial text.
    
    Features:
    - Entity extraction (companies, people, locations, etc.)
    - Sentiment analysis for financial texts
    - Company name to ticker symbol mapping
    - Text embedding generation for financial NLP tasks
    - Batch processing capabilities for efficient analysis
    - ONNX optimization for improved inference performance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Financial Text MCP server.

        Args:
            config: Optional configuration dictionary. If None, the config will be loaded
                  from the default path: config/financial_text_mcp/financial_text_mcp_config.json
        """
        # Load config from default path if not provided
        if config is None:
            config_path = os.path.join("config", "financial_text_mcp", "financial_text_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    # Initialize logger first for proper error handling
                    self.logger = NetdataLogger(component_name="financial-text-mcp")
                    self.logger.error(f"Error loading config from {config_path}: {e}")
                    config = {}
            else:
                # Initialize logger first for proper error handling
                self.logger = NetdataLogger(component_name="financial-text-mcp")
                self.logger.warning(f"Config file not found at {config_path}. Using default settings.")
                config = {}

        # Initialize monitoring components
        if not hasattr(self, 'logger'):
            self.logger = NetdataLogger(component_name="financial-text-mcp")
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.metrics_collector.start()

        # Initialize base MCP server
        super().__init__(name="financial_text_mcp", config=config)

        # Initialize cache and locks
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._model_load_lock = threading.RLock()
        self._last_model_use = time.time()
        
        # Extract configuration values
        self._configure_from_config()

        # Create cache directory if needed
        if self.enable_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load models
        self._load_models()

        # Register tools
        self._register_tools()
        
        # Set up health check
        self._last_health_check = time.time()
        self._health_status = {"status": "healthy", "models_loaded": self._models_loaded}

        # Register additional tools for monitoring and management
        self._register_management_tools()
        
        self.logger.info("Financial Text MCP initialized successfully")
        
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
            "models_loaded": self._health_status["models_loaded"],
            "uptime_seconds": time.time() - self._last_model_use,
            "cache_size": len(self._cache),
            "last_updated": self._health_status.get("last_updated", datetime.now().isoformat())
        }
        
    def _update_health_status(self) -> None:
        """Update the health status of the MCP server."""
        self._last_health_check = time.time()
        
        # Check if models are loaded
        models_loaded = {
            "ner": self.ner_pipeline is not None,
            "sentiment": (self.sentiment_pipeline is not None or self.sentiment_ort_session is not None),
            "embeddings": (self.embeddings_model is not None and self.embeddings_tokenizer is not None)
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
        with self._model_load_lock:
            self.ner_pipeline = None
            self.sentiment_pipeline = None
            self.sentiment_ort_session = None
            self.sentiment_tokenizer = None
            self.embeddings_model = None
            self.embeddings_tokenizer = None
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            if self.use_gpu and HAVE_TRANSFORMERS and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Load models again
        self._load_models()
        
        # Return status
        processing_time = time.time() - start_time
        return {
            "success": self._models_loaded,
            "models_loaded": self._health_status["models_loaded"],
            "status": self._health_status["status"],
            "processing_time": processing_time
        }

    def _configure_from_config(self):
        """Extract configuration values from the config dictionary."""
        # Model configurations
        models_config = self.config.get("models", {})

        # NER model config
        ner_config = models_config.get("ner", {})
        self.ner_model_path = ner_config.get("model_path", "ProsusAI/finbert-ner")
        self.ner_quantization = ner_config.get("quantization_enabled", True)

        # Sentiment model config
        sentiment_config = models_config.get("sentiment", {})
        self.sentiment_model_path = sentiment_config.get("model_path", "yiyanghkust/finbert-tone")
        self.sentiment_use_onnx = sentiment_config.get("use_onnx", HAVE_ONNX)

        # Embeddings model config
        embeddings_config = models_config.get("embeddings", {})
        self.embeddings_model_path = embeddings_config.get("model_path", "yiyanghkust/bert-fin")
        self.embedding_dimension = embeddings_config.get("embedding_dimension", 768)

        # General settings
        general_settings = self.config.get("general_settings", {})
        self.cache_dir = general_settings.get("cache_dir", "./model_cache")
        self.enable_cache = general_settings.get("enable_cache", True)
        self.cache_ttl = general_settings.get("cache_ttl", 3600)  # Default 1 hour cache TTL
        self.model_load_retries = general_settings.get("model_load_retries", 2)

        # Performance settings
        performance_config = self.config.get("performance", {})
        self.batch_size = performance_config.get("batch_size", 16)
        self.max_sequence_length = performance_config.get("max_sequence_length", 512)
        self.use_gpu = performance_config.get("use_gpu", torch.cuda.is_available() if HAVE_TRANSFORMERS else False)
        self.model_unload_threshold = performance_config.get("model_unload_threshold", 3600)  # Unload after 1 hour of inactivity

        # Company names and ticker mappings
        self.companies = self.config.get("companies", [])
        self.ticker_to_name = self.config.get("ticker_to_name", {})
        self.name_to_ticker = {name.lower(): ticker for ticker, name in self.ticker_to_name.items()}

        # Log configuration
        self.logger.info("Configuration loaded",
                       ner_model=self.ner_model_path,
                       sentiment_model=self.sentiment_model_path,
                       embeddings_model=self.embeddings_model_path,
                       use_gpu=self.use_gpu,
                       batch_size=self.batch_size,
                       cache_enabled=self.enable_cache)

    def _load_models(self):
        """Load all required pre-trained models with retry logic."""
        with self._model_load_lock:
            self._models_loaded = False
            self._last_model_use = time.time()
            
            if not HAVE_TRANSFORMERS:
                self.logger.warning("Transformers not available. Using fallback implementations.")
                self.ner_pipeline = None
                self.sentiment_pipeline = None
                self.sentiment_ort_session = None
                self.embeddings_model = None
                self.embeddings_tokenizer = None
                return

            device = 0 if self.use_gpu else -1
            models_loaded = {"ner": False, "sentiment": False, "embeddings": False}
            
            # Load NER model with retries
            for attempt in range(self.model_load_retries + 1):
                try:
                    self.logger.info(f"Loading NER model: {self.ner_model_path} (attempt {attempt+1}/{self.model_load_retries+1})")
                    start_time = time.time()
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=self.ner_model_path,
                        tokenizer=self.ner_model_path,
                        cache_dir=self.cache_dir,
                        device=device
                    )
                    load_time = time.time() - start_time
                    self.logger.timing("model_load_time_ms.ner", load_time * 1000)
                    self.logger.info(f"NER model loaded in {load_time:.2f}s")
                    models_loaded["ner"] = True
                    break
                except Exception as e:
                    self.logger.error(f"Error loading NER model (attempt {attempt+1}): {e}", exc_info=True)
                    if attempt < self.model_load_retries:
                        self.logger.info("Retrying NER model load in 2 seconds...")
                        time.sleep(2)
                    else:
                        self.logger.error(f"Failed to load NER model after {self.model_load_retries+1} attempts")
                        self.ner_pipeline = None

            # Load sentiment model with retries
            for attempt in range(self.model_load_retries + 1):
                try:
                    self.logger.info(f"Loading sentiment model: {self.sentiment_model_path} (attempt {attempt+1}/{self.model_load_retries+1})")
                    start_time = time.time()
                    self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                        self.sentiment_model_path, cache_dir=self.cache_dir
                    )

                    if self.sentiment_use_onnx and HAVE_ONNX:
                        onnx_path = os.path.join(
                            self.cache_dir, f"{self.sentiment_model_path.replace('/', '_')}.onnx"
                        )
                        if not os.path.exists(onnx_path):
                            self._export_sentiment_to_onnx(onnx_path)

                        self.logger.info(f"Loading ONNX sentiment model from {onnx_path}")
                        sess_options = ort.SessionOptions()
                        if self.ner_quantization:
                            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
                        self.sentiment_ort_session = ort.InferenceSession(
                            onnx_path, sess_options=sess_options, providers=providers
                        )
                        self._using_onnx = True
                        self.sentiment_pipeline = None
                    else:
                        self.sentiment_pipeline = pipeline(
                            "sentiment-analysis",
                            model=self.sentiment_model_path,
                            tokenizer=self.sentiment_tokenizer,
                            cache_dir=self.cache_dir,
                            device=device
                        )
                        self._using_onnx = False
                        self.sentiment_ort_session = None

                    load_time = time.time() - start_time
                    self.logger.timing("model_load_time_ms.sentiment", load_time * 1000)
                    self.logger.info(f"Sentiment model loaded in {load_time:.2f}s")
                    models_loaded["sentiment"] = True
                    break
                except Exception as e:
                    self.logger.error(f"Error loading sentiment model (attempt {attempt+1}): {e}", exc_info=True)
                    if attempt < self.model_load_retries:
                        self.logger.info("Retrying sentiment model load in 2 seconds...")
                        time.sleep(2)
                    else:
                        self.logger.error(f"Failed to load sentiment model after {self.model_load_retries+1} attempts")
                        self.sentiment_pipeline = None
                        self.sentiment_ort_session = None
                        self.sentiment_tokenizer = None

            # Load embeddings model with retries
            for attempt in range(self.model_load_retries + 1):
                try:
                    self.logger.info(f"Loading embeddings model: {self.embeddings_model_path} (attempt {attempt+1}/{self.model_load_retries+1})")
                    start_time = time.time()
                    self.embeddings_tokenizer = AutoTokenizer.from_pretrained(
                        self.embeddings_model_path, cache_dir=self.cache_dir
                    )
                    # Use AutoModel instead of AutoModelForSequenceClassification for better embeddings
                    self.embeddings_model = AutoModel.from_pretrained(
                        self.embeddings_model_path, cache_dir=self.cache_dir
                    )
                    if self.use_gpu:
                        self.embeddings_model.to('cuda')

                    load_time = time.time() - start_time
                    self.logger.timing("model_load_time_ms.embeddings", load_time * 1000)
                    self.logger.info(f"Embeddings model loaded in {load_time:.2f}s")
                    models_loaded["embeddings"] = True
                    break
                except Exception as e:
                    self.logger.error(f"Error loading embeddings model (attempt {attempt+1}): {e}", exc_info=True)
                    if attempt < self.model_load_retries:
                        self.logger.info("Retrying embeddings model load in 2 seconds...")
                        time.sleep(2)
                    else:
                        self.logger.error(f"Failed to load embeddings model after {self.model_load_retries+1} attempts")
                        self.embeddings_model = None
                        self.embeddings_tokenizer = None

            # Update models_loaded status
            self._models_loaded = any(models_loaded.values())
            self._health_status = {
                "status": "healthy" if self._models_loaded else "degraded",
                "models_loaded": models_loaded,
                "last_updated": datetime.now().isoformat()
            }
            
            if self._models_loaded:
                self.logger.info(f"Models loaded successfully: {models_loaded}")
            else:
                self.logger.warning("No models were loaded successfully. Using fallback implementations.")

    def _export_sentiment_to_onnx(self, onnx_path: str) -> bool:
        """
        Exports the sentiment model to ONNX format.
        
        Args:
            onnx_path: Path where the ONNX model will be saved
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        self.logger.info(f"Exporting sentiment model to ONNX: {onnx_path}")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_path, cache_dir=self.cache_dir
            )
            dummy_input = self.sentiment_tokenizer(
                "ONNX export test", return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_length
            )
            input_names = list(dummy_input.keys())
            output_names = ["logits"]

            model.eval()
            
            # Export with dynamic batch size
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={name: {0: "batch_size"} for name in input_names},
                opset_version=12,
                do_constant_folding=True,
            )
            
            # Verify the model
            if HAVE_ONNX:
                try:
                    ort_session = ort.InferenceSession(onnx_path)
                    ort_inputs = {k: v.numpy() for k, v in dummy_input.items()}
                    ort_session.run(None, ort_inputs)
                    self.logger.info("ONNX model verification successful")
                except Exception as verify_err:
                    self.logger.error(f"ONNX model verification failed: {verify_err}", exc_info=True)
                    return False
            
            self.logger.info(f"Model successfully exported to ONNX: {onnx_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export sentiment model to ONNX: {e}", exc_info=True)
            # Fallback to PyTorch if export fails
            self.sentiment_use_onnx = False
            return False


    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            self.extract_entities,
            "extract_entities",
            "Extract entities (companies, people, locations, etc.) from text"
        )
        self.register_tool(
            self.map_to_tickers,
            "map_to_tickers",
            "Map company names to ticker symbols"
        )
        self.register_tool(
            self.analyze_sentiment,
            "analyze_sentiment",
            "Analyze sentiment of a text input"
        )
        self.register_tool(
            self.batch_analyze_sentiment,
            "batch_analyze_sentiment",
            "Analyze sentiment of multiple text inputs in a batch"
        )
        self.register_tool(
            self.analyze_financial_text,
            "analyze_financial_text",
            "Comprehensive analysis of financial text (entities + sentiment)"
        )
        self.register_tool(
            self.analyze_news,
            "analyze_news",
            "Analyze a news article with title and content"
        )
        self.register_tool(
            self.generate_embeddings,
            "generate_embeddings",
            "Generate embeddings for text using financial language models"
        )

    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract entities from text."""
        start_time = time.time()
        if not text:
            self.logger.warning("Empty text provided for entity extraction")
            return {"entities": {}, "processing_time": 0.0}
            
        # Check cache first
        cache_key_params = {"text": text, "entity_types": entity_types}
        cached_result = self._get_from_cache("extract_entities", **cache_key_params)
        if cached_result is not None:
            self.logger.info("Using cached entity extraction result")
            cached_result["from_cache"] = True
            cached_result["processing_time"] = 0.0
            return cached_result

        self.logger.info("Extracting entities", text_length=len(text), entity_types=str(entity_types))
        entities_result = {}
        self._last_model_use = time.time()

        try:
            if not self._models_loaded or self.ner_pipeline is None:
                self.logger.warning("Using fallback entity extraction (NER model not loaded)")
                entities_result = self._extract_entities_fallback(text)
            else:
                ner_results = self.ner_pipeline(text)
                entities_result = self._process_ner_results(text, ner_results, entity_types)
                self._identify_companies(text, entities_result)

            processing_time = time.time() - start_time
            entity_count = sum(len(v) for v in entities_result.values())
            self.logger.timing(f"entity_extraction_time_ms.text_{len(text)}_entities_{entity_count}", processing_time * 1000)
            for entity_type, entities in entities_result.items():
                self.logger.gauge(f"entity_count_{entity_type}", len(entities))
                
            result = {"entities": entities_result, "processing_time": processing_time}
            
            # Cache the result
            self._add_to_cache("extract_entities", result, **cache_key_params)
            
            return result

        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return {"entities": {}, "error": str(e), "processing_time": processing_time}

    def _extract_entities_fallback(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Simple fallback entity extraction using regex patterns."""
        entities = {"COMPANY": [], "TICKER": []}
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        for match in re.finditer(ticker_pattern, text):
            ticker = match.group(0)
            if ticker in self.ticker_to_name:
                entities["TICKER"].append({
                    "text": ticker, "start": match.start(), "end": match.end(),
                    "company": self.ticker_to_name.get(ticker, "")
                })

        for company, ticker in self.name_to_ticker.items():
            try:
                # Use word boundaries for company names
                company_pattern = r'\b' + re.escape(company) + r'\b'
                for match in re.finditer(company_pattern, text, re.IGNORECASE):
                    start, end = match.span()
                    original_text = text[start:end]
                    # Avoid adding duplicates if already found as ORG/COMPANY by NER
                    is_duplicate = False
                    if "COMPANY" in entities:
                         for existing_entity in entities["COMPANY"]:
                             if existing_entity["start"] == start and existing_entity["end"] == end:
                                 is_duplicate = True
                                 break
                    if not is_duplicate:
                        entities["COMPANY"].append({
                            "text": original_text, "start": start, "end": end, "ticker": ticker
                        })
            except re.error as re_err:
                 self.logger.warning(f"Regex error for company '{company}': {re_err}")

        return entities

    def _process_ner_results(self, text: str, ner_results: List[Dict[str, Any]],
                           entity_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Process NER results from Hugging Face pipeline."""
        entities = {}
        for item in ner_results:
            entity_type = item['entity_group']
            if entity_types and entity_type not in entity_types:
                continue

            if entity_type not in entities:
                entities[entity_type] = []

            entity_info = {
                "text": item['word'],
                "start": item['start'],
                "end": item['end'],
                "confidence": round(item['score'], 4)
            }

            # Add ticker symbol for company entities if possible
            if entity_type in ('ORG', 'COMPANY') and item['word'].lower() in self.name_to_ticker:
                entity_info["ticker"] = self.name_to_ticker[item['word'].lower()]

            entities[entity_type].append(entity_info)
        return entities


    def _add_entity_to_results(self, entities: Dict[str, List[Dict[str, Any]]],
                             entity_type: str, entity_text: str,
                             start: int, end: int, original_text: str) -> None:
        """Helper to add an entity to the results dictionary."""
        if entity_type not in entities:
            entities[entity_type] = []

        original_entity = original_text[start:end]
        entity_info = {"text": original_entity, "start": start, "end": end}

        if entity_type in ('ORG', 'COMPANY') and original_entity.lower() in self.name_to_ticker:
            entity_info["ticker"] = self.name_to_ticker[original_entity.lower()]

        # Avoid adding duplicates
        if entity_info not in entities[entity_type]:
            entities[entity_type].append(entity_info)

    def map_to_tickers(self, text: str) -> Dict[str, Any]:
        """
        Map company names in the input text to ticker symbols.

        Args:
            text: Input text containing company names.

        Returns:
            Dict with mapping results.
        """
        if not text:
            return {"tickers": [], "mapping": {}, "error": None}

        found = {}
        tickers = []
        lowered = text.lower()
        for name, ticker in self.name_to_ticker.items():
            if name in lowered:
                found[name] = ticker
                tickers.append(ticker)
        return {"tickers": tickers, "mapping": found, "error": None}

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text input.

        Args:
            text: Input text.

        Returns:
            Dict with sentiment result.
        """
        start_time = time.time()
        
        if not text:
            return {"sentiment": "neutral", "score": 0.5, "processing_time": 0.0}
            
        # Check cache first
        cache_key_params = {"text": text}
        cached_result = self._get_from_cache("analyze_sentiment", **cache_key_params)
        if cached_result is not None:
            self.logger.info("Using cached sentiment analysis result")
            cached_result["from_cache"] = True
            cached_result["processing_time"] = 0.0
            return cached_result

        self.logger.info("Analyzing sentiment", text_length=len(text))
        self._last_model_use = time.time()

        try:
            if not self._models_loaded or (self.sentiment_pipeline is None and self.sentiment_ort_session is None):
                self.logger.warning("Using fallback sentiment analysis (model not loaded)")
                # Simple lexicon-based fallback
                positive_words = ["good", "great", "excellent", "positive", "profit", "increase", "up", "growth", "better"]
                negative_words = ["bad", "terrible", "poor", "negative", "loss", "decrease", "down", "worse", "risk"]
                
                text_lower = text.lower()
                pos_count = sum(word in text_lower for word in positive_words)
                neg_count = sum(word in text_lower for word in negative_words)
                
                total = pos_count + neg_count
                if total == 0:
                    sentiment = "neutral"
                    score = 0.5
                elif pos_count > neg_count:
                    sentiment = "positive"
                    score = 0.5 + (pos_count / (pos_count + neg_count)) / 2
                else:
                    sentiment = "negative"
                    score = 0.5 - (neg_count / (pos_count + neg_count)) / 2
                    
                result = {"sentiment": sentiment, "score": round(score, 4)}
            
            elif self._using_onnx and self.sentiment_ort_session is not None:
                # Use ONNX model
                inputs = self.sentiment_tokenizer(
                    text, return_tensors="np", padding=True, truncation=True,
                    max_length=self.max_sequence_length
                )
                ort_inputs = {k: v for k, v in inputs.items()}
                logits = self.sentiment_ort_session.run(None, ort_inputs)[0]
                
                # Apply softmax to get probabilities
                import numpy as np
                probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                
                # Get predicted class and score
                prediction = np.argmax(probabilities, axis=1)[0]
                score = probabilities[0][prediction]
                
                # Map to sentiment labels (adjust based on model's label order)
                labels = ["negative", "neutral", "positive"]
                sentiment = labels[prediction]
                
                result = {"sentiment": sentiment, "score": float(score)}
                
            elif self.sentiment_pipeline is not None:
                # Use Hugging Face pipeline
                pipeline_result = self.sentiment_pipeline(text)[0]
                
                label = pipeline_result['label']
                score = pipeline_result['score']
                
                # Map to standard sentiment values
                if 'positive' in label.lower():
                    sentiment = 'positive'
                elif 'negative' in label.lower():
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                    
                result = {"sentiment": sentiment, "score": float(score)}
            
            else:
                self.logger.error("No sentiment analysis method available")
                result = {"sentiment": "neutral", "score": 0.5, "error": "No sentiment analysis method available"}

            processing_time = time.time() - start_time
            self.logger.timing(f"sentiment_analysis_time_ms.text_{len(text)}", processing_time * 1000)
            result["processing_time"] = processing_time
            
            # Cache the result
            self._add_to_cache("analyze_sentiment", result, **cache_key_params)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return {"sentiment": "neutral", "score": 0.5, "error": str(e), "processing_time": processing_time}
        
    def batch_analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple text inputs in a batch.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Dict with sentiment results for all texts.
        """
        start_time = time.time()
        
        if not texts:
            return {"results": [], "processing_time": 0.0}
            
        # Process each text individually
        results = []
        for text in texts:
            # Reuse the single-text sentiment analysis
            sentiment_result = self.analyze_sentiment(text)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "score": sentiment_result.get("score", 0.0)
            })
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "processing_time": processing_time,
            "count": len(results)
        }
        
    def analyze_financial_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of financial text (entities + sentiment).
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dict with combined entity and sentiment analysis.
        """
        start_time = time.time()
        
        if not text:
            return {
                "entities": {},
                "sentiment": "neutral",
                "score": 0.0,
                "processing_time": 0.0
            }
            
        # Get entities first
        entities_result = self.extract_entities(text)
        
        # Get sentiment
        sentiment_result = self.analyze_sentiment(text)
        
        # Combine the results
        result = {
            "entities": entities_result.get("entities", {}),
            "sentiment": sentiment_result.get("sentiment", "neutral"),
            "score": sentiment_result.get("score", 0.0),
            "processing_time": time.time() - start_time,
        }
        
        # Add ticker symbols if present
        tickers = self.map_to_tickers(text)
        if tickers.get("tickers"):
            result["tickers"] = tickers.get("tickers", [])
            result["company_mapping"] = tickers.get("mapping", {})
            
        self.logger.info("Completed financial text analysis",
                       text_length=len(text),
                       entity_count=sum(len(v) for v in result["entities"].values()),
                       sentiment=result["sentiment"])
        
        return result
        
    def analyze_news(self, title: str, content: str) -> Dict[str, Any]:
        """
        Analyze a news article with title and content.
        
        Args:
            title: Article title
            content: Article content/body
            
        Returns:
            Dict with analysis results including sentiment and entities
        """
        start_time = time.time()
        
        if not title and not content:
            return {
                "error": "Both title and content are empty",
                "processing_time": 0.0
            }
            
        # Combine title and content for analysis, giving more weight to the title
        full_text = f"{title} {title} {content}" if title else content
        
        # Get comprehensive analysis of the combined text
        analysis_result = self.analyze_financial_text(full_text)
        
        # Extract title sentiment separately if title is provided
        title_sentiment = None
        if title:
            title_sentiment_result = self.analyze_sentiment(title)
            title_sentiment = {
                "sentiment": title_sentiment_result.get("sentiment", "neutral"),
                "score": title_sentiment_result.get("score", 0.0)
            }
        
        # Build the complete result
        result = {
            "title": title,
            "content_summary": content[:100] + "..." if len(content) > 100 else content,
            "overall_sentiment": analysis_result.get("sentiment", "neutral"),
            "sentiment_score": analysis_result.get("score", 0.0),
            "title_sentiment": title_sentiment,
            "entities": analysis_result.get("entities", {}),
            "tickers": analysis_result.get("tickers", []),
            "processing_time": time.time() - start_time
        }
        
        self.logger.info("Completed news article analysis",
                       title_length=len(title) if title else 0,
                       content_length=len(content) if content else 0,
                       sentiment=result["overall_sentiment"])
        
        return result
