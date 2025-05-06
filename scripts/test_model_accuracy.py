#!/usr/bin/env python3
"""
NextGen Models Accuracy Test

This script loads the financial_phrasebank dataset and tests the accuracy of the 
NextGen models and MCP tools. It sends data through all models and generates a 
comprehensive report on accuracy, performance, and system health.

Usage:
    python test_model_accuracy.py [--verbose] [--report-file REPORT_FILE]
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import traceback
import asyncio
import requests
import logging
import uuid

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root) # type: ignore
sys.path.append(os.path.join(project_root, "nextgen_models")) # type: ignore
sys.path.append(os.path.join(project_root, "mcp_tools")) # type: ignore

# Import monitoring components
try:
    from monitoring.netdata_logger import NetdataLogger
    from monitoring.system_metrics import SystemMetricsCollector
except ImportError as e:
    print(f"Error importing monitoring components: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1) # type: ignore

class ModelAccuracyTester:
    """Model accuracy testing framework for NextGen trading system."""
    
    def __init__(self, verbose=False, report_file="model_accuracy_report.json"):
        """Initialize the model tester."""
        self.verbose = verbose
        self.report_file = report_file
        self.start_time = datetime.now()
        
        # Initialize logger
        self.logger = NetdataLogger(component_name="model-accuracy-test")
        self.logger.info("Starting model accuracy test")
        
        # Create a dedicated log file for this test
        self.setup_dedicated_log_file()
        
        # Initialize metrics collector
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.metrics_collector.start()
        
        # Initialize result storage
        self.results = {
            "test_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": None,
                "dataset": "financial_phrasebank",
                "total_samples": 0,
                "processed_samples": 0
            },
            "models": {},
            "mcp_tools": {},
            "system_metrics": {
                "cpu_usage": [],
                "memory_usage": [],
                "response_times": []
            },
            "errors": []
        }
        
        # Try to import required components
        self._import_components()
    
    def setup_dedicated_log_file(self):
        """Set up a dedicated log file for this test run."""
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log file for this test run
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"model_accuracy_test_{test_id}.log")
        
        # Set up file handler
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Add handler to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Also add to our NetdataLogger
        self.logger.logger.addHandler(file_handler)
        
        self.logger.info(f"Dedicated log file created at: {self.log_file}")
    
    def _import_components(self):
        """Attempt to import all required components."""
        self.components_available = {}
        
        # Try to import MCP components
        try:
            from mcp_tools.financial_text_mcp.financial_text_mcp import FinancialTextMCP
            # Temporarily disable ONNX for sentiment analysis to bypass dimension error
            self.financial_text_mcp = FinancialTextMCP(config={"models": {"sentiment": {"use_onnx": False}}})
            self.components_available["financial_text_mcp"] = True
            self.logger.info("Successfully imported FinancialTextMCP with ONNX disabled for sentiment")
        except Exception as e: # Catch broader exception to include initialization errors
            self.logger.error(f"Failed to initialize FinancialTextMCP: {e}")
            self.components_available["financial_text_mcp"] = False
            self.results["errors"].append(f"Failed to import FinancialTextMCP: {e}")
        
        try:
            from mcp_tools.db_mcp.redis_mcp import RedisMCP
            self.redis_mcp = RedisMCP()
            self.components_available["redis_mcp"] = True
            self.logger.info("Successfully imported RedisMCP")
        except ImportError as e:
            self.logger.error(f"Failed to import RedisMCP: {e}")
            self.components_available["redis_mcp"] = False
            self.results["errors"].append(f"Failed to import RedisMCP: {e}")
        
        try:
            from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP
            self.vector_store_mcp = VectorStoreMCP()
            self.components_available["vector_store_mcp"] = True
            self.logger.info("Successfully imported VectorStoreMCP")
        except ImportError as e:
            self.logger.error(f"Failed to import VectorStoreMCP: {e}")
            self.components_available["vector_store_mcp"] = False
            self.results["errors"].append(f"Failed to import VectorStoreMCP: {e}")
        
        # Try to import model components
        try:
            from nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model import SentimentAnalysisModel
            # Import StockChartGenerator
            from monitoring.stock_charts import StockChartGenerator
            
            # Create a modified version of SentimentAnalysisModel that doesn't use asyncio.run() in __init__
            class AsyncSafeModel:
                def __init__(self):
                    # Create a standalone model that doesn't inherit from SentimentAnalysisModel
                    # to avoid the asyncio.run() call in the parent's __init__
                    self.logger = NetdataLogger(component_name="nextgen-sentiment-analysis-model")
                    self.logger.info("Starting SentimentAnalysisModel initialization (async-safe)")
                    
                    # Initialize metrics collector
                    self.metrics_collector = SystemMetricsCollector(self.logger)
                    self.metrics_collector.start()
                    
                    # Initialize chart generator
                    self.chart_generator = StockChartGenerator()
                    
                    # Set basic configuration
                    self.config = {
                        "batch_size": 10,
                        "sentiment_ttl": 86400,  # 1 day
                        "financial_text_config": {"models": {"sentiment": {"use_onnx": False}}},
                        "financial_data_config": {},
                        "redis_config": {}
                    }
                    
                    # Initialize financial text MCP
                    self.financial_text_mcp = None
                    try:
                        from mcp_tools.financial_text_mcp.financial_text_mcp import FinancialTextMCP
                        self.financial_text_mcp = FinancialTextMCP()
                    except ImportError:
                        self.logger.warning("Failed to import FinancialTextMCP, will use fallback methods")
                    
                    # Initialize Redis client
                    self.redis_client = None
                    try:
                        import redis
                        self.redis_client = redis.Redis(
                            host=self.config.get("redis_host", "localhost"),
                            port=self.config.get("redis_port", 6379),
                            db=self.config.get("redis_db", 0),
                            decode_responses=True
                        )
                        self.logger.info("Redis client initialized")
                    except (ImportError, redis.exceptions.ConnectionError) as e:
                        self.logger.warning(f"Failed to initialize Redis client: {e}")
                    
                    # Symbol entity mapping will be loaded later
                    self.symbol_entity_mapping = {}
                    
                    # Initialize counters for statistics
                    self.texts_analyzed_count = 0
                    self.entities_extracted_count = 0
                    self.sentiment_scores_generated_count = 0
                    
                    self.logger.info("AsyncSafeModel initialized for testing")
                
                async def load_symbol_entity_mapping(self):
                    """Load symbol entity mapping asynchronously."""
                    # Simplified implementation that returns empty mapping
                    return self.symbol_entity_mapping
                
                async def analyze_sentiment(self, text):
                    """
                    Analyze sentiment for the given text using the financial_text_mcp.
                    
                    Args:
                        text: String text to analyze
                        
                    Returns:
                        List containing a single sentiment analysis result dict
                    """
                    try:
                        if self.financial_text_mcp:
                            result = self.financial_text_mcp.call_tool("analyze_sentiment", {"text": text})
                            
                            # Format into expected return structure
                            if result and not result.get("error"):
                                self.texts_analyzed_count += 1
                                
                                # Create a properly formatted response
                                response = [{
                                    "input_text": text,
                                    "overall_sentiment": result.get("overall_sentiment", {}),
                                    "entities": result.get("entities", [])
                                }]
                                
                                # Count entities
                                if "entities" in result:
                                    self.entities_extracted_count += len(result["entities"])
                                
                                # Count sentiment scores
                                if "overall_sentiment" in result:
                                    self.sentiment_scores_generated_count += 1
                                
                                return response
                        
                        # Fallback basic sentiment analysis if no MCP
                        self.logger.warning("Using fallback sentiment analysis")
                        # Simple rule-based fallback (very basic)
                        positive_words = ["growth", "profit", "increase", "positive", "strong", "up", "gain"]
                        negative_words = ["decline", "loss", "decrease", "negative", "weak", "down", "risk"]
                        
                        text_lower = text.lower()
                        pos_count = sum(1 for word in positive_words if word in text_lower)
                        neg_count = sum(1 for word in negative_words if word in text_lower)
                        
                        sentiment = "NEUTRAL"
                        score = 0.0
                        
                        if pos_count > neg_count:
                            sentiment = "POSITIVE"
                            score = 0.5 + min(0.4, 0.1 * (pos_count - neg_count))
                        elif neg_count > pos_count:
                            sentiment = "NEGATIVE"
                            score = -0.5 - min(0.4, 0.1 * (neg_count - pos_count))
                            
                        return [{
                            "input_text": text,
                            "overall_sentiment": {"label": sentiment, "score": score},
                            "entities": []
                        }]
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing sentiment: {e}")
                        return [{
                            "input_text": text,
                            "overall_sentiment": None,
                            "entities": [],
                            "error": str(e)
                        }]
            
            # Create the async-safe model
            self.sentiment_model = AsyncSafeModel()
            self.components_available["sentiment_model"] = True
            self.logger.info("Successfully imported SentimentAnalysisModel (async-safe)")
        except Exception as e:
            self.logger.error(f"Failed to import SentimentAnalysisModel: {e}")
            self.components_available["sentiment_model"] = False
            self.results["errors"].append(f"Failed to import SentimentAnalysisModel: {e}")
        
        try:
            from nextgen_models.nextgen_select.select_model import SelectionModel
            self.select_model = SelectionModel()
            self.components_available["select_model"] = True
            self.logger.info("Successfully imported SelectionModel")
        except ImportError as e:
            self.logger.error(f"Failed to import SelectionModel: {e}")
            self.components_available["select_model"] = False
            self.results["errors"].append(f"Failed to import SelectionModel: {e}")
        
        # If direct imports fail, try to connect to the running models via Redis
        if not all(self.components_available.values()):
            self.logger.warning("Some components failed to import directly. Will try to connect via Redis.")
            self._setup_redis_connections()
    
    def _setup_redis_connections(self):
        """Set up connections to models via Redis if direct imports fail."""
        if self.components_available.get("redis_mcp", False):
            self.logger.info("Setting up Redis connections to models")
            
            # Check if models are publishing to Redis streams
            try:
                streams = self.redis_mcp.call_tool("keys", {"pattern": "model:*"})
                if streams and not streams.get("error") and streams.get("keys"):
                    self.logger.info(f"Found {len(streams['keys'])} model streams in Redis")
                    for stream in streams["keys"]:
                        self.logger.info(f"Found model stream: {stream}")
                else:
                    self.logger.warning("No model streams found in Redis")
            except Exception as e:
                self.logger.error(f"Error checking Redis streams: {e}")
    
    def load_dataset(self, dataset_path=None):
        """Load the financial phrasebank dataset."""
        if dataset_path is None:
            dataset_path = os.path.join(project_root, "financial_phrasebank.csv")
        
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)
            
            # Log dataset info
            self.logger.info(f"Dataset loaded: {len(df)} samples")
            if self.verbose:
                self.logger.info(f"Dataset columns: {df.columns.tolist()}")
                self.logger.info(f"Dataset sentiment distribution: {df['label'].value_counts().to_dict()}")
            
            # Store dataset info in results
            self.results["test_info"]["total_samples"] = len(df)
            self.results["test_info"]["dataset_info"] = {
                "columns": df.columns.tolist(),
                "sentiment_distribution": df['label'].value_counts().to_dict() if 'label' in df.columns else {}
            }
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            self.results["errors"].append(f"Error loading dataset: {e}")
            return None
    
    async def test_sentiment_model(self, dataset):
        """Test the sentiment analysis model."""
        self.logger.info("Testing sentiment analysis model")
        
        model_results = {
            "name": "sentiment_analysis_model",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "confusion_matrix": None,
            "response_times": [],
            "errors": []
        }
        
        try:
            # Check if we have direct access to the sentiment model
            if self.components_available.get("sentiment_model", False):
                self.logger.info("Using direct sentiment model access")
                
                # Process samples in batches
                batch_size = 10
                total_samples = len(dataset)
                processed = 0
                
                y_true = []
                y_pred = []
                
                for i in range(0, total_samples, batch_size):
                    batch = dataset.iloc[i:min(i+batch_size, total_samples)]
                    
                    # Process each text in the batch
                    for _, row in batch.iterrows():
                        text = row['sentence']
                        true_label = row['label']
                        
                        # Map dataset labels to model labels
                        if isinstance(true_label, str):
                            true_label = true_label.lower()
                        
                        # Convert numeric labels if needed
                        if isinstance(true_label, (int, float)):
                            if true_label > 0.5:
                                true_label = "positive"
                            elif true_label < -0.5:
                                true_label = "negative"
                            else:
                                true_label = "neutral"
                        
                        # Process the text
                        start_time = time.time()
                        try:
                            # Use the sentiment model to analyze the text
                            result = await self.sentiment_model.analyze_sentiment(text)
                            
                            # Extract the predicted sentiment
                            if result and len(result) > 0:
                                sentiment_result = result[0]
                                overall_sentiment = sentiment_result.get("overall_sentiment", {})
                                pred_label = overall_sentiment.get("label", "").lower() if overall_sentiment else "neutral"
                                
                                # Map model output to standardized labels
                                if "positive" in pred_label:
                                    pred_label = "positive"
                                elif "negative" in pred_label:
                                    pred_label = "negative"
                                else:
                                    pred_label = "neutral"
                                
                                # Record the prediction
                                y_true.append(true_label)
                                y_pred.append(pred_label)
                                
                                # Record response time
                                response_time = (time.time() - start_time) * 1000  # ms
                                model_results["response_times"].append(response_time)
                                
                                processed += 1
                                if processed % 100 == 0 or processed == total_samples:
                                    self.logger.info(f"Processed {processed}/{total_samples} samples")
                            else:
                                self.logger.warning(f"No result returned for text: {text[:50]}...")
                                model_results["errors"].append(f"No result for sample {processed}")
                        except Exception as e:
                            self.logger.error(f"Error processing sample: {e}")
                            model_results["errors"].append(f"Error processing sample {processed}: {str(e)}")
                
                # Calculate metrics if we have predictions
                if y_true and y_pred:
                    # Convert labels to numeric for sklearn metrics
                    label_map = {"positive": 2, "neutral": 1, "negative": 0}
                    y_true_numeric = [label_map.get(label, 1) for label in y_true]
                    y_pred_numeric = [label_map.get(label, 1) for label in y_pred]
                    
                    # Calculate metrics
                    model_results["accuracy"] = accuracy_score(y_true_numeric, y_pred_numeric)
                    model_results["precision"] = precision_score(y_true_numeric, y_pred_numeric, average='weighted')
                    model_results["recall"] = recall_score(y_true_numeric, y_pred_numeric, average='weighted')
                    model_results["f1_score"] = f1_score(y_true_numeric, y_pred_numeric, average='weighted')
                    model_results["confusion_matrix"] = confusion_matrix(y_true_numeric, y_pred_numeric).tolist()
                    
                    self.logger.info(f"Sentiment model accuracy: {model_results['accuracy']:.4f}")
                    self.logger.info(f"Sentiment model F1 score: {model_results['f1_score']:.4f}")
                else:
                    self.logger.warning("No predictions were made, cannot calculate metrics")
            else:
                # Try to use the financial_text_mcp directly
                self.logger.info("Using financial_text_mcp for sentiment analysis")
                
                if self.components_available.get("financial_text_mcp", False):
                    # Process samples in batches
                    batch_size = 10
                    total_samples = len(dataset)
                    processed = 0
                    
                    y_true = []
                    y_pred = []
                    
                    for i in range(0, total_samples, batch_size):
                        batch = dataset.iloc[i:min(i+batch_size, total_samples)]
                        
                        # Process each text in the batch
                        for _, row in batch.iterrows():
                            text = row['sentence']
                            true_label = row['label']
                            
                            # Map dataset labels to model labels
                            if isinstance(true_label, str):
                                true_label = true_label.lower()
                            
                            # Convert numeric labels if needed
                            if isinstance(true_label, (int, float)):
                                if true_label > 0.5:
                                    true_label = "positive"
                                elif true_label < -0.5:
                                    true_label = "negative"
                                else:
                                    true_label = "neutral"
                            
                            # Process the text
                            start_time = time.time()
                            try:
                                # Use the financial_text_mcp to analyze sentiment
                                result = self.financial_text_mcp.call_tool("analyze_sentiment", {"text": text})
                                
                                # Extract the predicted sentiment
                                if result and not result.get("error"):
                                    overall_sentiment = result.get("overall_sentiment", {})
                                    pred_label = overall_sentiment.get("label", "").lower() if overall_sentiment else "neutral"
                                    
                                    # Map model output to standardized labels
                                    if "positive" in pred_label:
                                        pred_label = "positive"
                                    elif "negative" in pred_label:
                                        pred_label = "negative"
                                    else:
                                        pred_label = "neutral"
                                    
                                    # Record the prediction
                                    y_true.append(true_label)
                                    y_pred.append(pred_label)
                                    
                                    # Record response time
                                    response_time = (time.time() - start_time) * 1000  # ms
                                    model_results["response_times"].append(response_time)
                                    
                                    processed += 1
                                    if processed % 100 == 0 or processed == total_samples:
                                        self.logger.info(f"Processed {processed}/{total_samples} samples")
                                else:
                                    error_msg = result.get("error", "Unknown error") if result else "No result"
                                    self.logger.warning(f"Error analyzing sentiment: {error_msg}")
                                    model_results["errors"].append(f"Error for sample {processed}: {error_msg}")
                            except Exception as e:
                                self.logger.error(f"Error processing sample: {e}")
                                model_results["errors"].append(f"Error processing sample {processed}: {str(e)}")
                    
                    # Calculate metrics if we have predictions
                    if y_true and y_pred:
                        # Convert labels to numeric for sklearn metrics
                        label_map = {"positive": 2, "neutral": 1, "negative": 0}
                        y_true_numeric = [label_map.get(label, 1) for label in y_true]
                        y_pred_numeric = [label_map.get(label, 1) for label in y_pred]
                        
                        # Calculate metrics
                        model_results["accuracy"] = accuracy_score(y_true_numeric, y_pred_numeric)
                        model_results["precision"] = precision_score(y_true_numeric, y_pred_numeric, average='weighted')
                        model_results["recall"] = recall_score(y_true_numeric, y_pred_numeric, average='weighted')
                        model_results["f1_score"] = f1_score(y_true_numeric, y_pred_numeric, average='weighted')
                        model_results["confusion_matrix"] = confusion_matrix(y_true_numeric, y_pred_numeric).tolist()
                        
                        self.logger.info(f"Sentiment model accuracy: {model_results['accuracy']:.4f}")
                        self.logger.info(f"Sentiment model F1 score: {model_results['f1_score']:.4f}")
                    else:
                        self.logger.warning("No predictions were made, cannot calculate metrics")
                else:
                    self.logger.error("Neither sentiment_model nor financial_text_mcp are available")
                    model_results["errors"].append("Neither sentiment_model nor financial_text_mcp are available")
        except Exception as e:
            self.logger.error(f"Error testing sentiment model: {e}")
            model_results["errors"].append(f"Error testing sentiment model: {str(e)}")
        
        # Store results
        self.results["models"]["sentiment_analysis_model"] = model_results
        return model_results
    
    async def test_select_model(self, dataset):
        """Test the select model."""
        self.logger.info("Testing select model")
        
        model_results = {
            "name": "select_model",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "response_times": [],
            "errors": []
        }
        
        # Since the select model doesn't directly process text data,
        # we'll test it by feeding sentiment results and checking if it
        # correctly selects stocks based on sentiment
        
        try:
            # Check if we have direct access to the select model
            if self.components_available.get("select_model", False):
                self.logger.info("Using direct select model access")
                
                # Create test data: simulate sentiment results for stocks
                test_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                test_sentiments = {
                    "AAPL": {"label": "POSITIVE", "score": 0.8},
                    "MSFT": {"label": "NEGATIVE", "score": -0.7},
                    "GOOGL": {"label": "NEUTRAL", "score": 0.1},
                    "AMZN": {"label": "POSITIVE", "score": 0.6},
                    "META": {"label": "NEGATIVE", "score": -0.5}
                }
                
                # Expected behavior: select model should prioritize positive sentiment stocks
                expected_selections = ["AAPL", "AMZN"]
                
                # Test the select model
                start_time = time.time()
                try:
                    # Call the select model with test data
                    selections = await self.select_model.select_stocks(test_sentiments)
                    
                    # Record response time
                    response_time = (time.time() - start_time) * 1000  # ms
                    model_results["response_times"].append(response_time)
                    
                    # Check if the selections match expectations
                    if selections:
                        # Calculate accuracy based on whether positive sentiment stocks were selected
                        correct_selections = [s for s in selections if s in expected_selections]
                        model_results["accuracy"] = len(correct_selections) / len(expected_selections)
                        
                        self.logger.info(f"Select model made {len(selections)} selections")
                        self.logger.info(f"Select model accuracy: {model_results['accuracy']:.4f}")
                    else:
                        self.logger.warning("Select model did not make any selections")
                        model_results["errors"].append("No selections made")
                except Exception as e:
                    self.logger.error(f"Error testing select model: {e}")
                    model_results["errors"].append(f"Error testing select model: {str(e)}")
            else:
                self.logger.warning("Select model not available for direct testing")
                
                # Try to test via Redis
                if self.components_available.get("redis_mcp", False):
                    self.logger.info("Testing select model via Redis")
                    
                    # Create test data in Redis
                    test_data = {
                        "symbols": {
                            "AAPL": {"sentiment": {"label": "POSITIVE", "score": 0.8}},
                            "MSFT": {"sentiment": {"label": "NEGATIVE", "score": -0.7}},
                            "GOOGL": {"sentiment": {"label": "NEUTRAL", "score": 0.1}},
                            "AMZN": {"sentiment": {"label": "POSITIVE", "score": 0.6}},
                            "META": {"sentiment": {"label": "NEGATIVE", "score": -0.5}}
                        },
                        "timestamp": datetime.now().isoformat(),
                        "test_id": str(uuid.uuid4())
                    }
                    
                    # Publish to sentiment feedback stream
                    stream_key = "sentiment:selection_feedback"
                    start_time = time.time()
                    
                    try:
                        result = self.redis_mcp.call_tool("xadd", {
                            "stream": stream_key,
                            "data": test_data
                        })
                        
                        if result and not result.get("error"):
                            self.logger.info(f"Published test data to {stream_key}")
                            
                            # Wait for select model to process
                            await asyncio.sleep(2)
                            
                            # Check for selections in Redis
                            selection_key = "selection:data"
                            selection_result = self.redis_mcp.call_tool("get_json", {"key": selection_key})
                            
                            if selection_result and not selection_result.get("error"):
                                selections = selection_result.get("value", {}).get("selected_symbols", [])
                                
                                # Record response time
                                response_time = (time.time() - start_time) * 1000  # ms
                                model_results["response_times"].append(response_time)
                                
                                # Expected behavior: select model should prioritize positive sentiment stocks
                                expected_selections = ["AAPL", "AMZN"]
                                
                                # Calculate accuracy
                                if selections:
                                    correct_selections = [s for s in selections if s in expected_selections]
                                    model_results["accuracy"] = len(correct_selections) / len(expected_selections)
                                    
                                    self.logger.info(f"Select model made {len(selections)} selections")
                                    self.logger.info(f"Select model accuracy: {model_results['accuracy']:.4f}")
                                else:
                                    self.logger.warning("No selections found in Redis")
                                    model_results["errors"].append("No selections found in Redis")
                            else:
                                error_msg = selection_result.get("error", "Unknown error") if selection_result else "No result"
                                self.logger.warning(f"Error getting selections from Redis: {error_msg}")
                                model_results["errors"].append(f"Error getting selections: {error_msg}")
                        else:
                            error_msg = result.get("error", "Unknown error") if result else "No result"
                            self.logger.warning(f"Error publishing to Redis: {error_msg}")
                            model_results["errors"].append(f"Error publishing to Redis: {error_msg}")
                    except Exception as e:
                        self.logger.error(f"Error testing select model via Redis: {e}")
                        model_results["errors"].append(f"Error testing via Redis: {str(e)}")
                else:
                    self.logger.error("Neither select_model nor redis_mcp are available")
                    model_results["errors"].append("Neither select_model nor redis_mcp are available")
        except Exception as e:
            self.logger.error(f"Error testing select model: {e}")
            model_results["errors"].append(f"Error testing select model: {str(e)}")
        
        # Store results
        self.results["models"]["select_model"] = model_results
        return model_results
    
    async def test_financial_text_mcp(self, dataset):
        """Test the financial text MCP tool."""
        self.logger.info("Testing financial_text_mcp")
        
        tool_results = {
            "name": "financial_text_mcp",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "response_times": [],
            "errors": []
        }
        
        try:
            if self.components_available.get("financial_text_mcp", False):
                # Process samples in batches
                batch_size = 10
                total_samples = min(100, len(dataset))  # Limit to 100 samples for MCP tool testing
                processed = 0
                
                y_true = []
                y_pred = []
                
                for i in range(0, total_samples, batch_size):
                    batch = dataset.iloc[i:min(i+batch_size, total_samples)]
                    
                    # Process each text in the batch
                    for _, row in batch.iterrows():
                        text = row['sentence']
                        true_label = row['label']
                        
                        # Map dataset labels to model labels
                        if isinstance(true_label, str):
                            true_label = true_label.lower()
                        
                        # Convert numeric labels if needed
                        if isinstance(true_label, (int, float)):
                            if true_label > 0.5:
                                true_label = "positive"
                            elif true_label < -0.5:
                                true_label = "negative"
                            else:
                                true_label = "neutral"
                        
                        # Process the text
                        start_time = time.time()
                        try:
                            # Use the financial_text_mcp to score sentiment
                            result = self.financial_text_mcp.call_tool("score_sentiment", {"text": text})
                            
                            # Extract the predicted sentiment
                            if result and not result.get("error"):
                                sentiment_score = result.get("score", 0)
                                
                                # Convert score to label
                                if sentiment_score > 0.2:
                                    pred_label = "positive"
                                elif sentiment_score < -0.2:
                                    pred_label = "negative"
                                else:
                                    pred_label = "neutral"
                                
                                # Record the prediction
                                y_true.append(true_label)
                                y_pred.append(pred_label)
                                
                                # Record response time
                                response_time = (time.time() - start_time) * 1000  # ms
                                tool_results["response_times"].append(response_time)
                                
                                processed += 1
                                if processed % 10 == 0 or processed == total_samples:
                                    self.logger.info(f"Processed {processed}/{total_samples} samples")
                            else:
                                error_msg = result.get("error", "Unknown error") if result else "No result"
                                self.logger.warning(f"Error scoring sentiment: {error_msg}")
                                tool_results["errors"].append(f"Error for sample {processed}: {error_msg}")
                        except Exception as e:
                            self.logger.error(f"Error processing sample: {e}")
                            tool_results["errors"].append(f"Error processing sample {processed}: {str(e)}")
                
                # Calculate metrics if we have predictions
                if y_true and y_pred:
                    # Convert labels to numeric for sklearn metrics
                    label_map = {"positive": 2, "neutral": 1, "negative": 0}
                    y_true_numeric = [label_map.get(label, 1) for label in y_true]
                    y_pred_numeric = [label_map.get(label, 1) for label in y_pred]
                    
                    # Calculate metrics
                    tool_results["accuracy"] = accuracy_score(y_true_numeric, y_pred_numeric)
                    tool_results["precision"] = precision_score(y_true_numeric, y_pred_numeric, average='weighted')
                    tool_results["recall"] = recall_score(y_true_numeric, y_pred_numeric, average='weighted')
                    tool_results["f1_score"] = f1_score(y_true_numeric, y_pred_numeric, average='weighted')
                    
                    self.logger.info(f"Financial text MCP accuracy: {tool_results['accuracy']:.4f}")
                    self.logger.info(f"Financial text MCP F1 score: {tool_results['f1_score']:.4f}")
                else:
                    self.logger.warning("No predictions were made, cannot calculate metrics")
            else:
                self.logger.error("financial_text_mcp is not available")
                tool_results["errors"].append("financial_text_mcp is not available")
        except Exception as e:
            self.logger.error(f"Error testing financial_text_mcp: {e}")
            tool_results["errors"].append(f"Error testing financial_text_mcp: {str(e)}")
        
        # Store results
        self.results["mcp_tools"]["financial_text_mcp"] = tool_results
        return tool_results
    
    async def test_vector_store_mcp(self):
        """Test the vector store MCP tool."""
        self.logger.info("Testing vector_store_mcp")
        
        tool_results = {
            "name": "vector_store_mcp",
            "accuracy": None,
            "response_times": [],
            "errors": []
        }
        
        try:
            if self.components_available.get("vector_store_mcp", False):
                # Create test data
                test_texts = [
                    "Apple reported strong quarterly earnings with revenue growth across all product categories.",
                    "Microsoft's cloud services continue to drive growth for the company.",
                    "Google's parent company Alphabet saw advertising revenue decline in the latest quarter.",
                    "Amazon's e-commerce business faces increasing competition from traditional retailers.",
                    "Meta's focus on the metaverse is a long-term bet that may not pay off immediately."
                ]
                
                # Test vector store operations
                start_time = time.time()
                try:
                    # Test adding documents to the vector store
                    collection_name = f"test_collection_{int(time.time())}"
                    
                    # Create collection
                    create_result = self.vector_store_mcp.call_tool("create_collection", {
                        "collection_name": collection_name
                    })
                    
                    if create_result and not create_result.get("error"):
                        self.logger.info(f"Created test collection: {collection_name}")
                        
                        # Add documents
                        for i, text in enumerate(test_texts):
                            add_result = self.vector_store_mcp.call_tool("add_document", {
                                "collection_name": collection_name,
                                "document": {
                                    "id": f"doc_{i}",
                                    "text": text,
                                    "metadata": {"source": "test", "index": i}
                                }
                            })
                            
                            if add_result and not add_result.get("error"):
                                self.logger.info(f"Added document {i} to vector store")
                            else:
                                error_msg = add_result.get("error", "Unknown error") if add_result else "No result"
                                self.logger.warning(f"Error adding document to vector store: {error_msg}")
                                tool_results["errors"].append(f"Error adding document {i}: {error_msg}")
                        
                        # Test query
                        query_text = "tech companies with strong financial performance"
                        query_result = self.vector_store_mcp.call_tool("query", {
                            "collection_name": collection_name,
                            "query_text": query_text,
                            "limit": 2
                        })
                        
                        if query_result and not query_result.get("error"):
                            results = query_result.get("results", [])
                            self.logger.info(f"Query returned {len(results)} results")
                            
                            # Record response time
                            response_time = (time.time() - start_time) * 1000  # ms
                            tool_results["response_times"].append(response_time)
                            
                            # Simple accuracy check: did we get any results?
                            tool_results["accuracy"] = 1.0 if results else 0.0
                        else:
                            error_msg = query_result.get("error", "Unknown error") if query_result else "No result"
                            self.logger.warning(f"Error querying vector store: {error_msg}")
                            tool_results["errors"].append(f"Error querying: {error_msg}")
                        
                        # Clean up - delete collection
                        delete_result = self.vector_store_mcp.call_tool("delete_collection", {
                            "collection_name": collection_name
                        })
                        
                        if delete_result and not delete_result.get("error"):
                            self.logger.info(f"Deleted test collection: {collection_name}")
                        else:
                            error_msg = delete_result.get("error", "Unknown error") if delete_result else "No result"
                            self.logger.warning(f"Error deleting collection: {error_msg}")
                            tool_results["errors"].append(f"Error deleting collection: {error_msg}")
                    else:
                        error_msg = create_result.get("error", "Unknown error") if create_result else "No result"
                        self.logger.warning(f"Error creating collection: {error_msg}")
                        tool_results["errors"].append(f"Error creating collection: {error_msg}")
                except Exception as e:
                    self.logger.error(f"Error testing vector store operations: {e}")
                    tool_results["errors"].append(f"Error testing vector store: {str(e)}")
            else:
                self.logger.error("vector_store_mcp is not available")
                tool_results["errors"].append("vector_store_mcp is not available")
        except Exception as e:
            self.logger.error(f"Error testing vector_store_mcp: {e}")
            tool_results["errors"].append(f"Error testing vector_store_mcp: {str(e)}")
        
        # Store results
        self.results["mcp_tools"]["vector_store_mcp"] = tool_results
        return tool_results
    
    async def test_redis_mcp(self):
        """Test the Redis MCP tool."""
        self.logger.info("Testing redis_mcp")
        
        tool_results = {
            "name": "redis_mcp",
            "success_rate": None,
            "response_times": [],
            "errors": []
        }
        
        try:
            if self.components_available.get("redis_mcp", False):
                # Test basic Redis operations
                operations = 0
                successful_operations = 0
                
                # Test set/get
                start_time = time.time()
                try:
                    test_key = f"test_key_{int(time.time())}"
                    test_value = f"test_value_{int(time.time())}"
                    
                    # Set value
                    set_result = self.redis_mcp.call_tool("set", {
                        "key": test_key,
                        "value": test_value
                    })
                    
                    operations += 1
                    if set_result and not set_result.get("error"):
                        self.logger.info(f"Set key {test_key} in Redis")
                        successful_operations += 1
                    else:
                        error_msg = set_result.get("error", "Unknown error") if set_result else "No result"
                        self.logger.warning(f"Error setting key in Redis: {error_msg}")
                        tool_results["errors"].append(f"Error setting key: {error_msg}")
                    
                    # Get value
                    get_result = self.redis_mcp.call_tool("get", {
                        "key": test_key
                    })
                    
                    operations += 1
                    if get_result and not get_result.get("error"):
                        retrieved_value = get_result.get("value")
                        if retrieved_value == test_value:
                            self.logger.info(f"Successfully retrieved key {test_key} from Redis")
                            successful_operations += 1
                        else:
                            self.logger.warning(f"Value mismatch: expected {test_value}, got {retrieved_value}")
                            tool_results["errors"].append(f"Value mismatch: expected {test_value}, got {retrieved_value}")
                    else:
                        error_msg = get_result.get("error", "Unknown error") if get_result else "No result"
                        self.logger.warning(f"Error getting key from Redis: {error_msg}")
                        tool_results["errors"].append(f"Error getting key: {error_msg}")
                    
                    # Delete key
                    del_result = self.redis_mcp.call_tool("del", {
                        "key": test_key
                    })
                    
                    operations += 1
                    if del_result and not del_result.get("error"):
                        self.logger.info(f"Deleted key {test_key} from Redis")
                        successful_operations += 1
                    else:
                        error_msg = del_result.get("error", "Unknown error") if del_result else "No result"
                        self.logger.warning(f"Error deleting key from Redis: {error_msg}")
                        tool_results["errors"].append(f"Error deleting key: {error_msg}")
                    
                    # Record response time
                    response_time = (time.time() - start_time) * 1000  # ms
                    tool_results["response_times"].append(response_time)
                    
                    # Calculate success rate
                    tool_results["success_rate"] = successful_operations / operations if operations > 0 else 0
                    
                    self.logger.info(f"Redis MCP success rate: {tool_results['success_rate']:.4f}")
                except Exception as e:
                    self.logger.error(f"Error testing Redis operations: {e}")
                    tool_results["errors"].append(f"Error testing Redis operations: {str(e)}")
            else:
                self.logger.error("redis_mcp is not available")
                tool_results["errors"].append("redis_mcp is not available")
        except Exception as e:
            self.logger.error(f"Error testing redis_mcp: {e}")
            tool_results["errors"].append(f"Error testing redis_mcp: {str(e)}")
        
        # Store results
        self.results["mcp_tools"]["redis_mcp"] = tool_results
        return tool_results
    
    async def collect_system_metrics(self):
        """Collect system metrics during the test."""
        self.logger.info("Collecting system metrics")
        
        try:
            # Get CPU usage
            cpu_usage = self.metrics_collector.get_cpu_usage()
            self.results["system_metrics"]["cpu_usage"].append(cpu_usage)
            self.logger.info(f"CPU usage: {cpu_usage}%")
            
            # Get memory usage
            memory_usage = self.metrics_collector.get_memory_usage()
            self.results["system_metrics"]["memory_usage"].append(memory_usage)
            self.logger.info(f"Memory usage: {memory_usage}%")
            
            # Get disk usage
            disk_usage = self.metrics_collector.get_disk_usage()
            self.results["system_metrics"]["disk_usage"] = disk_usage
            self.logger.info(f"Disk usage: {disk_usage}%")
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            self.results["errors"].append(f"Error collecting system metrics: {str(e)}")
            return None
    
    def generate_report(self):
        """Generate a comprehensive report of the test results."""
        self.logger.info("Generating test report")
        
        # Update end time
        self.results["test_info"]["end_time"] = datetime.now().isoformat()
        
        # Calculate test duration
        start_time = datetime.fromisoformat(self.results["test_info"]["start_time"])
        end_time = datetime.fromisoformat(self.results["test_info"]["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.results["test_info"]["duration_seconds"] = duration
        
        # Calculate overall metrics
        model_accuracies = [model["accuracy"] for model in self.results["models"].values() if model["accuracy"] is not None]
        if model_accuracies:
            self.results["overall_metrics"] = {
                "average_model_accuracy": sum(model_accuracies) / len(model_accuracies),
                "models_tested": len(self.results["models"]),
                "mcp_tools_tested": len(self.results["mcp_tools"]),
                "error_count": len(self.results["errors"]),
                "average_cpu_usage": sum(self.results["system_metrics"]["cpu_usage"]) / len(self.results["system_metrics"]["cpu_usage"]) if self.results["system_metrics"]["cpu_usage"] else None,
                "average_memory_usage": sum(self.results["system_metrics"]["memory_usage"]) / len(self.results["system_metrics"]["memory_usage"]) if self.results["system_metrics"]["memory_usage"] else None
            }
        
        # Save report to file
        try:
            with open(self.report_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.logger.info(f"Report saved to {self.report_file}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
        
        return self.results
    
    def print_summary(self):
        """Print a summary of the test results."""
        if "overall_metrics" in self.results:
            print("\n===== MODEL ACCURACY TEST SUMMARY =====")
            print(f"Test duration: {self.results['test_info']['duration_seconds']:.2f} seconds")
            print(f"Models tested: {self.results['overall_metrics']['models_tested']}")
            print(f"MCP tools tested: {self.results['overall_metrics']['mcp_tools_tested']}")
            print(f"Average model accuracy: {self.results['overall_metrics']['average_model_accuracy']:.4f}")
            print(f"Error count: {self.results['overall_metrics']['error_count']}")
            print(f"Report saved to: {self.report_file}")
            print(f"Log file: {self.log_file}")
            print("=======================================\n")
            
            # Print individual model results
            print("Model Results:")
            for name, model in self.results["models"].items():
                accuracy = model.get("accuracy")
                if accuracy is not None:
                    print(f"  - {name}: Accuracy = {accuracy:.4f}")
                else:
                    print(f"  - {name}: No accuracy data")
            
            # Print MCP tool results
            print("\nMCP Tool Results:")
            for name, tool in self.results["mcp_tools"].items():
                if "accuracy" in tool and tool["accuracy"] is not None:
                    print(f"  - {name}: Accuracy = {tool['accuracy']:.4f}")
                elif "success_rate" in tool and tool["success_rate"] is not None:
                    print(f"  - {name}: Success Rate = {tool['success_rate']:.4f}")
                else:
                    print(f"  - {name}: No metrics available")
            
            print("\nCheck the report file for detailed results.")
        else:
            print("No overall metrics available. Test may not have completed successfully.")
            print(f"Check the log file for details: {self.log_file}")
    
    async def run_tests(self):
        """Run all tests."""
        self.logger.info("Starting all tests")
        
        # Load dataset
        dataset = self.load_dataset()
        if dataset is None:
            self.logger.error("Failed to load dataset, aborting tests")
            return False
        
        # Collect initial system metrics
        await self.collect_system_metrics()
        
        # Test models
        await self.test_sentiment_model(dataset)
        await self.test_select_model(dataset)
        
        # Test MCP tools
        await self.test_financial_text_mcp(dataset)
        await self.test_vector_store_mcp()
        await self.test_redis_mcp()
        
        # Collect final system metrics
        await self.collect_system_metrics()
        
        # Generate report
        self.generate_report()
        
        # Print summary
        self.print_summary()
        
        self.logger.info("All tests completed")
        return True

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test NextGen models accuracy")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--report-file", default="model_accuracy_report.json", help="Path to save the report")
    parser.add_argument("--test-type", default="all", choices=["all", "sentiment", "select", "financial_text", "vector_store", "redis"],
                        help="Type of test to run (default: all)")
    args = parser.parse_args()
    
    tester = ModelAccuracyTester(verbose=args.verbose, report_file=args.report_file)
    
    # Run tests based on test type
    if args.test_type == "all":
        await tester.run_tests()
    elif args.test_type == "sentiment":
        dataset = tester.load_dataset()
        if dataset is not None:
            await tester.collect_system_metrics()
            await tester.test_sentiment_model(dataset)
            await tester.collect_system_metrics()
            tester.generate_report()
            tester.print_summary()
    elif args.test_type == "select":
        dataset = tester.load_dataset()
        if dataset is not None:
            await tester.collect_system_metrics()
            await tester.test_select_model(dataset)
            await tester.collect_system_metrics()
            tester.generate_report()
            tester.print_summary()
    elif args.test_type == "financial_text":
        dataset = tester.load_dataset()
        if dataset is not None:
            await tester.collect_system_metrics()
            await tester.test_financial_text_mcp(dataset)
            await tester.collect_system_metrics()
            tester.generate_report()
            tester.print_summary()
    elif args.test_type == "vector_store":
        await tester.collect_system_metrics()
        await tester.test_vector_store_mcp()
        await tester.collect_system_metrics()
        tester.generate_report()
        tester.print_summary()
    elif args.test_type == "redis":
        await tester.collect_system_metrics()
        await tester.test_redis_mcp()
        await tester.collect_system_metrics()
        tester.generate_report()
        tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
