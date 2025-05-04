"""
Sentiment Scoring MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
sentiment analysis capabilities for financial text using FinBERT and other models.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional

# For ONNX optimization
try:
    import onnxruntime as ort

    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

# For PyTorch models
try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# For tokenization and preprocessing
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-sentiment-scoring",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class SentimentScoringMCP(BaseMCPServer):
    """
    MCP server for financial sentiment analysis using specialized models like FinBERT.

    This tool processes financial text (news, social media, filings) and provides
    sentiment scores with confidence ratings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Sentiment Scoring MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - model_path: Path to FinBERT or other sentiment model
                - use_onnx: Whether to use ONNX runtime for optimization
                - use_quantization: Whether to use INT8 quantization
                - batch_size: Size of batches for processing
                - cache_dir: Directory for caching models
        """
        super().__init__(name="sentiment_scoring_mcp", config=config)

        # Set default configurations if not provided
        self.model_path = self.config.get(
            "model_path",
            "yiyanghkust/finbert-tone",  # Default to FinBERT
        )
        self.use_onnx = self.config.get("use_onnx", HAVE_ONNX)
        self.use_quantization = self.config.get("use_quantization", True)
        self.batch_size = self.config.get("batch_size", 16)
        self.cache_dir = self.config.get("cache_dir", "./model_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize tokenizer and model
        self._init_model()

        # Register tools
        self._register_tools()

    def _init_model(self):
        """Initialize the sentiment analysis model."""
        if not HAVE_TRANSFORMERS:
            self.logger.warning(
                "Hugging Face Transformers not available. Some functionality will be limited."
            )
            self._model_loaded = False
            return

        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, cache_dir=self.cache_dir
            )

            # Initialize model - either with ONNX or standard PyTorch
            if self.use_onnx and HAVE_ONNX:
                # Export model to ONNX if not already exported
                onnx_path = os.path.join(
                    self.cache_dir, f"{self.model_path.replace('/', '_')}.onnx"
                )

                if not os.path.exists(onnx_path):
                    # Check for PyTorch availability
                    if not HAVE_TORCH:
                        self.logger.error(
                            "PyTorch is required for model export to ONNX"
                        )
                        raise ImportError(
                            "PyTorch is required for model export to ONNX"
                        )

                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_path, cache_dir=self.cache_dir
                    )

                    # Export model to ONNX format
                    self.logger.info(f"Exporting model to ONNX format at {onnx_path}")

                    # Create dummy inputs for tracing
                    dummy_input = self.tokenizer(
                        "This is a dummy input for ONNX export",
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )

                    # Get input names
                    input_names = list(dummy_input.keys())
                    # Common output names for transformer models
                    output_names = ["logits"]

                    # Export to ONNX using torch.onnx.export
                    import torch.onnx

                    # Set model to evaluation mode
                    model.eval()

                    # Export the model
                    torch.onnx.export(
                        model,  # PyTorch model
                        tuple(dummy_input.values()),  # Model inputs
                        onnx_path,  # Output file
                        input_names=input_names,  # Input names
                        output_names=output_names,  # Output names
                        dynamic_axes={  # Dynamic axes for variable batch size
                            input_name: {0: "batch_size"} for input_name in input_names
                        },
                        opset_version=12,  # ONNX opset version
                        do_constant_folding=True,  # Optimize model
                        verbose=False,
                    )

                    self.logger.info(f"Model successfully exported to {onnx_path}")

                    # Use ONNX runtime for inference
                    self._using_onnx = True
                else:
                    # Set up ONNX runtime
                    self.logger.info(f"Loading ONNX model from {onnx_path}")
                    sess_options = ort.SessionOptions()

                    # Configure session for performance
                    if self.use_quantization:
                        sess_options.graph_optimization_level = (
                            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        )

                    # Use CUDA provider if available
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self.ort_session = ort.InferenceSession(
                        onnx_path, sess_options=sess_options, providers=providers
                    )
                    self._using_onnx = True
            else:
                # Use standard Hugging Face model with PyTorch
                if not HAVE_TORCH:
                    self.logger.error(
                        "PyTorch is required for using the Hugging Face model"
                    )
                    raise ImportError(
                        "PyTorch is required for using the Hugging Face model"
                    )

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, cache_dir=self.cache_dir
                )
                self._using_onnx = False

            self._model_loaded = True
            self.logger.info(
                f"Successfully initialized sentiment model from {self.model_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment model: {e}")
            self._model_loaded = False

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "analyze_text",
            self.analyze_text,
            "Analyze sentiment of a text input",
            {
                "text": {
                    "type": "string",
                    "description": "Text to analyze for sentiment",
                }
            },
            {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "batch_analyze_texts",
            self.batch_analyze_texts,
            "Analyze sentiment of multiple text inputs in a batch",
            {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to analyze for sentiment",
                }
            },
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "sentiment": {"type": "string"},
                        "score": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        )

        self.register_tool(
            "analyze_news_sentiment",
            self.analyze_news_sentiment,
            "Analyze sentiment of a news article with title and content",
            {
                "title": {"type": "string", "description": "Title of the news article"},
                "content": {
                    "type": "string",
                    "description": "Content of the news article",
                },
                "source": {
                    "type": "string",
                    "description": "Source of the news article (optional)",
                    "required": False,
                },
                "date": {
                    "type": "string",
                    "description": "Publication date (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string"},
                    "score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "title_sentiment": {"type": "string"},
                    "content_sentiment": {"type": "string"},
                    "overall_sentiment": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "aggregate_sentiment",
            self.aggregate_sentiment,
            "Aggregate sentiment scores across multiple texts",
            {
                "sentiment_results": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of sentiment analysis results",
                },
                "weights": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional weights for each result",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "overall_sentiment": {"type": "string"},
                    "average_score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "positive_ratio": {"type": "number"},
                    "neutral_ratio": {"type": "number"},
                    "negative_ratio": {"type": "number"},
                },
            },
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        start_time = time.time()

        if not self._model_loaded:
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "error": "Model not loaded",
                "processing_time": 0.0,
            }

        try:
            # Tokenize the input
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )

            # Get prediction
            if self._using_onnx:
                # ONNX inference with proper implementation
                #
                # Convert PyTorch tensor inputs to numpy arrays for ONNX
                # Runtime
                ort_inputs = {name: value.numpy() for name, value in inputs.items()}

                # Run inference with ONNX Runtime
                ort_outputs = self.ort_session.run(
                    None,  # Output names (None = all outputs)
                    ort_inputs,
                )

                # Process outputs (usually the first output is the logits)
                logits = ort_outputs[0]  # Shape: [batch_size, num_classes]
                scores = logits[0]  # Get scores for the first (and only) input
            else:
                # PyTorch inference
                if not HAVE_TORCH:
                    raise ImportError(
                        "PyTorch is required for model inference but is not installed"
                    )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                scores = outputs.logits[0].numpy()

            #
            # Process results - assuming 3 classes: negative (0), neutral (1),
            # positive (2)
            sentiments = ["negative", "neutral", "positive"]
            sentiment_idx = np.argmax(scores)
            sentiment = sentiments[sentiment_idx]

            # Convert to probability with softmax
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Get score and confidence
            score = (
                sentiment_idx - 1
            ) + 0.5  # Map to [-0.5, 0.5, 1.5] then add 0.5 to get [0, 1, 2]
            confidence = float(probs[sentiment_idx])

            processing_time = time.time() - start_time

            return {
                "sentiment": sentiment,
                "score": float(score),
                "confidence": confidence,
                "processing_time": processing_time,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def batch_analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment analysis results
        """
        results = []

        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Process each text in the batch
            for text in batch:
                result = self.analyze_text(text)
                result["text"] = text[:100] + "..." if len(text) > 100 else text
                results.append(result)

        return results

    def analyze_news_sentiment(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of a news article with title and content.

        Args:
            title: Article title
            content: Article content
            source: News source (optional)
            date: Publication date (optional)

        Returns:
            Sentiment analysis results
        """
        # Analyze title sentiment
        title_result = self.analyze_text(title)

        # Analyze content sentiment (use first 1000 chars for speed)
        content_result = self.analyze_text(content[:1000])

        # Combine results (title sentiment has higher weight)
        title_weight = 0.4
        content_weight = 0.6

        weighted_score = (
            title_result["score"] * title_weight
            + content_result["score"] * content_weight
        )

        # Map score to sentiment category
        overall_sentiment = (
            "negative"
            if weighted_score < 0.4
            else "positive"
            if weighted_score > 0.6
            else "neutral"
        )

        # Calculate confidence based on individual confidences
        confidence = (
            title_result["confidence"] * title_weight
            + content_result["confidence"] * content_weight
        )

        return {
            "sentiment": overall_sentiment,
            "score": float(weighted_score),
            "confidence": float(confidence),
            "title_sentiment": title_result["sentiment"],
            "content_sentiment": content_result["sentiment"],
            "source": source,
            "date": date,
            "overall_sentiment": overall_sentiment,
        }

    def aggregate_sentiment(
        self,
        sentiment_results: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate sentiment scores across multiple results.

        Args:
            sentiment_results: List of sentiment analysis results
            weights: Optional weights for each result

        Returns:
            Aggregated sentiment metrics
        """
        if not sentiment_results:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.5,
                "confidence": 0.0,
                "positive_ratio": 0.0,
                "neutral_ratio": 0.0,
                "negative_ratio": 0.0,
            }

        # Use equal weights if not specified
        if weights is None:
            weights = [1.0] * len(sentiment_results)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(sentiment_results)
            total_weight = len(sentiment_results)

        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted scores and confidence
        weighted_scores = []
        weighted_confidences = []
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}

        for result, weight in zip(sentiment_results, normalized_weights):
            score = result.get("score", 0.5)
            confidence = result.get("confidence", 0.0)
            sentiment = result.get("sentiment", "neutral")

            weighted_scores.append(score * weight)
            weighted_confidences.append(confidence * weight)
            sentiments[sentiment] += weight

        # Calculate aggregate metrics
        average_score = sum(weighted_scores)
        confidence = sum(weighted_confidences)

        # Determine overall sentiment
        if average_score < 0.4:
            overall_sentiment = "negative"
        elif average_score > 0.6:
            overall_sentiment = "positive"
        else:
            overall_sentiment = "neutral"

        # Calculate sentiment ratios
        positive_ratio = sentiments["positive"]
        neutral_ratio = sentiments["neutral"]
        negative_ratio = sentiments["negative"]

        return {
            "overall_sentiment": overall_sentiment,
            "average_score": float(average_score),
            "confidence": float(confidence),
            "positive_ratio": float(positive_ratio),
            "neutral_ratio": float(neutral_ratio),
            "negative_ratio": float(negative_ratio),
        }


# Modules required for running this as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {
        "model_path": "yiyanghkust/finbert-tone",
        "use_onnx": True,
        "use_quantization": True,
        "batch_size": 16,
    }

    # Create and start the server
    server = SentimentScoringMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("SentimentScoringMCP server started")

    # Example usage
    result = server.analyze_text(
        "The company reported strong earnings, exceeding analyst expectations with a 15% growth in revenue."
    )
    print(f"Sentiment analysis result: {json.dumps(result, indent=2)}")
