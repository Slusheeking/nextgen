#!/usr/bin/env python3
"""
Test suite for the Embeddings MCP Tool.

This module tests the functionality of the EmbeddingsMCP class, including:
- Initialization and configuration
- Embedding generation for single texts
- Batch embedding generation
- Error handling
- Device selection
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the target module for testing
from mcp_tools.analysis_mcp.embeddings_mcp import EmbeddingsMCP


class TestEmbeddingsMCP(unittest.TestCase):
    """Test cases for the EmbeddingsMCP class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock for SentenceTransformer
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        
        # Configure the MCP server for testing
        config = {
            "model_name": "test-model",
            "use_gpu": False,
            "batch_size": 16,
            "cache_dir": "./test_embedding_cache"
        }
        
        # Create a mock for EmbeddingsMCP to avoid actual model loading
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.SentenceTransformer', return_value=self.mock_model), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.setup_monitoring', return_value=(MagicMock(), MagicMock())), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.EmbeddingsMCP._register_tools'):
            self.embeddings_mcp = EmbeddingsMCP(config)
            
        # Set up the model attribute directly
        self.embeddings_mcp.model = self.mock_model
        
        # Mock logger to avoid actual logging
        self.embeddings_mcp.logger = MagicMock()
        
        # Sample text data for testing
        self.sample_text = "This is a test sentence for embedding generation."
        self.sample_texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence."
        ]

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove test cache directory if it exists
        if os.path.exists("./test_embedding_cache"):
            import shutil
            shutil.rmtree("./test_embedding_cache")

    def test_initialization(self):
        """Test initialization with various configurations."""
        # Test default initialization
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.SentenceTransformer', return_value=self.mock_model), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.setup_monitoring', return_value=(MagicMock(), MagicMock())), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.EmbeddingsMCP._register_tools'):
            default_mcp = EmbeddingsMCP()
            
        # Verify default values
        self.assertEqual(default_mcp.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(default_mcp.batch_size, 32)
        
        # Test custom initialization
        custom_config = {
            "model_name": "custom-model",
            "batch_size": 64,
            "use_gpu": True
        }
        
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.SentenceTransformer', return_value=self.mock_model), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.setup_monitoring', return_value=(MagicMock(), MagicMock())), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.EmbeddingsMCP._register_tools'), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.EmbeddingsMCP._is_cuda_available', return_value=True):
            custom_mcp = EmbeddingsMCP(custom_config)
        
        # Verify custom values
        self.assertEqual(custom_mcp.model_name, "custom-model")
        self.assertEqual(custom_mcp.batch_size, 64)
        self.assertEqual(custom_mcp.use_gpu, True)
        self.assertEqual(custom_mcp.device, "cuda")

    def test_is_cuda_available(self):
        """Test CUDA availability check."""
        # Test when torch is available and CUDA is available
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.HAVE_TORCH', True), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.torch.cuda.is_available', return_value=True):
            self.assertTrue(self.embeddings_mcp._is_cuda_available())
        
        # Test when torch is available but CUDA is not available
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.HAVE_TORCH', True), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.torch.cuda.is_available', return_value=False):
            self.assertFalse(self.embeddings_mcp._is_cuda_available())
        
        # Test when torch is not available
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.HAVE_TORCH', False):
            self.assertFalse(self.embeddings_mcp._is_cuda_available())
        
        # Test when torch.cuda.is_available raises an exception
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.HAVE_TORCH', True), \
             patch('mcp_tools.analysis_mcp.embeddings_mcp.torch.cuda.is_available', side_effect=Exception("CUDA error")):
            self.assertFalse(self.embeddings_mcp._is_cuda_available())
            self.embeddings_mcp.logger.warning.assert_called_once()

    def test_init_model(self):
        """Test model initialization."""
        # Test successful model initialization
        mock_model = MagicMock()
        
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.SentenceTransformer', return_value=mock_model):
            self.embeddings_mcp._init_model()
            
        # Verify model was initialized
        self.assertEqual(self.embeddings_mcp.model, mock_model)
        
        # Test model initialization with CUDA
        self.embeddings_mcp.device = "cuda"
        mock_model = MagicMock()
        
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.SentenceTransformer', return_value=mock_model):
            self.embeddings_mcp._init_model()
            
        # Verify model was moved to CUDA
        mock_model.to.assert_called_once_with("cuda")
        
        # Test model initialization failure
        with patch('mcp_tools.analysis_mcp.embeddings_mcp.SentenceTransformer', side_effect=Exception("Model error")):
            self.embeddings_mcp._init_model()
            
        # Verify error was logged
        self.embeddings_mcp.logger.error.assert_called()
        
        # Verify model is None after failure
        self.assertIsNone(self.embeddings_mcp.model)

    def test_generate_embedding(self):
        """Test generate_embedding method."""
        # Set up mock to return a specific embedding
        embedding_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.mock_model.encode.return_value = np.array([embedding_array])
        
        # Test with valid input
        result = self.embeddings_mcp.generate_embedding(self.sample_text)
        
        # Verify structure of the result
        self.assertIn('embedding', result)
        self.assertIn('dimension', result)
        self.assertIn('model_name', result)
        self.assertIn('processing_time', result)
        
        # Verify the embedding values
        self.assertEqual(result['embedding'], embedding_array.tolist())
        self.assertEqual(result['dimension'], 5)
        self.assertEqual(result['model_name'], "test-model")
        
        # Verify model.encode was called correctly
        self.mock_model.encode.assert_called_with(
            [self.sample_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1
        )
        
        # Test with normalize=False
        self.mock_model.encode.reset_mock()
        result = self.embeddings_mcp.generate_embedding(self.sample_text, normalize=False)
        
        # Verify model.encode was called with normalize_embeddings=False
        self.mock_model.encode.assert_called_with(
            [self.sample_text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=1
        )

    def test_generate_embedding_with_empty_text(self):
        """Test generate_embedding with empty text."""
        # Test with empty text
        result = self.embeddings_mcp.generate_embedding("")
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embedding'])
        self.assertIsNone(result['dimension'])
        
        # Verify model.encode was not called
        self.mock_model.encode.assert_not_called()
        
        # Test with non-string input
        result = self.embeddings_mcp.generate_embedding(123)
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embedding'])
        self.assertIsNone(result['dimension'])

    def test_generate_embedding_with_model_error(self):
        """Test generate_embedding when model.encode raises an exception."""
        # Set up mock to raise an exception
        self.mock_model.encode.side_effect = Exception("Encoding error")
        
        # Test with valid input
        result = self.embeddings_mcp.generate_embedding(self.sample_text)
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embedding'])
        self.assertIsNone(result['dimension'])
        
        # Verify error was logged
        self.embeddings_mcp.logger.error.assert_called()

    def test_generate_embedding_with_no_model(self):
        """Test generate_embedding when model is None."""
        # Set model to None
        self.embeddings_mcp.model = None
        
        # Test with valid input
        result = self.embeddings_mcp.generate_embedding(self.sample_text)
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embedding'])
        self.assertIsNone(result['dimension'])
        self.assertEqual(result['error'], "Embedding model is not loaded.")

    def test_batch_generate_embeddings(self):
        """Test batch_generate_embeddings method."""
        # Set up mock to return specific embeddings
        embeddings_array = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ])
        self.mock_model.encode.return_value = embeddings_array
        
        # Test with valid input
        result = self.embeddings_mcp.batch_generate_embeddings(self.sample_texts)
        
        # Verify structure of the result
        self.assertIn('embeddings', result)
        self.assertIn('dimension', result)
        self.assertIn('model_name', result)
        self.assertIn('processing_time', result)
        
        # Verify the embedding values
        self.assertEqual(result['embeddings'], embeddings_array.tolist())
        self.assertEqual(result['dimension'], 5)
        self.assertEqual(result['model_name'], "test-model")
        
        # Verify model.encode was called correctly
        self.mock_model.encode.assert_called_with(
            self.sample_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False
        )
        
        # Test with normalize=False
        self.mock_model.encode.reset_mock()
        result = self.embeddings_mcp.batch_generate_embeddings(self.sample_texts, normalize=False)
        
        # Verify model.encode was called with normalize_embeddings=False
        self.mock_model.encode.assert_called_with(
            self.sample_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=16,
            show_progress_bar=False
        )

    def test_batch_generate_embeddings_with_empty_list(self):
        """Test batch_generate_embeddings with empty list."""
        # Test with empty list
        result = self.embeddings_mcp.batch_generate_embeddings([])
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embeddings'])
        self.assertIsNone(result['dimension'])
        
        # Verify model.encode was not called
        self.mock_model.encode.assert_not_called()
        
        # Test with non-list input
        result = self.embeddings_mcp.batch_generate_embeddings("not a list")
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embeddings'])
        self.assertIsNone(result['dimension'])

    def test_batch_generate_embeddings_with_non_string_items(self):
        """Test batch_generate_embeddings with non-string items in the list."""
        # Test with list containing non-string items
        result = self.embeddings_mcp.batch_generate_embeddings(["text", 123, "more text"])
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embeddings'])
        self.assertIsNone(result['dimension'])
        
        # Verify model.encode was not called
        self.mock_model.encode.assert_not_called()

    def test_batch_generate_embeddings_with_model_error(self):
        """Test batch_generate_embeddings when model.encode raises an exception."""
        # Set up mock to raise an exception
        self.mock_model.encode.side_effect = Exception("Batch encoding error")
        
        # Test with valid input
        result = self.embeddings_mcp.batch_generate_embeddings(self.sample_texts)
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embeddings'])
        self.assertIsNone(result['dimension'])
        
        # Verify error was logged
        self.embeddings_mcp.logger.error.assert_called()

    def test_batch_generate_embeddings_with_no_model(self):
        """Test batch_generate_embeddings when model is None."""
        # Set model to None
        self.embeddings_mcp.model = None
        
        # Test with valid input
        result = self.embeddings_mcp.batch_generate_embeddings(self.sample_texts)
        
        # Verify error handling
        self.assertIn('error', result)
        self.assertIsNone(result['embeddings'])
        self.assertIsNone(result['dimension'])
        self.assertEqual(result['error'], "Embedding model is not loaded.")

    def test_tool_registration(self):
        """Test that all required methods exist in EmbeddingsMCP class."""
        # Since we're mocking the tool registration process, test that the methods exist instead
        required_methods = {
            "generate_embedding",
            "batch_generate_embeddings"
        }
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.embeddings_mcp, method_name))
            self.assertTrue(callable(getattr(self.embeddings_mcp, method_name)))


if __name__ == "__main__":
    unittest.main()
