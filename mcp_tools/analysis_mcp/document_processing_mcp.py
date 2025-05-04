"""
Document Processing MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
document preprocessing capabilities for embedding generation and RAG systems.
"""

import os
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-document-processing",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class DocumentProcessingMCP(BaseMCPServer):
    """
    MCP server for preprocessing documents before embedding generation.

    This tool implements document chunking, metadata extraction, and cleaning
    to prepare documents for embedding generation and retrieval systems.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Document Processing MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - default_chunk_size: Default size for document chunks (default: 1000)
                - default_chunk_overlap: Default overlap between chunks (default: 200)
                - cache_dir: Directory for caching intermediate results
                - max_document_size: Maximum document size in bytes (default: 10MB)
                - supported_file_types: List of supported file types
        """
        super().__init__(name="document_processing_mcp", config=config)

        # Set default configurations
        self.default_chunk_size = self.config.get("default_chunk_size", 1000)
        self.default_chunk_overlap = self.config.get("default_chunk_overlap", 200)
        self.cache_dir = self.config.get("cache_dir", "./document_cache")
        self.max_document_size = self.config.get(
            "max_document_size", 10 * 1024 * 1024
        )  # 10MB
        self.supported_file_types = self.config.get(
            "supported_file_types", ["txt", "pdf", "docx", "html", "md"]
        )

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "chunk_document",
            self.chunk_document,
            "Split a document into chunks for embedding generation",
            {
                "document": {
                    "type": "string",
                    "description": "Document text to be chunked",
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Size of each chunk in characters",
                    "default": self.default_chunk_size,
                },
                "chunk_overlap": {
                    "type": "integer",
                    "description": "Overlap between consecutive chunks in characters",
                    "default": self.default_chunk_overlap,
                },
                "chunk_strategy": {
                    "type": "string",
                    "description": "Strategy for chunking: 'character', 'word', 'sentence', or 'paragraph'",
                    "default": "paragraph",
                },
            },
            {
                "type": "object",
                "properties": {
                    "chunks": {"type": "array"},  # List of document chunks
                    "chunk_metadata": {"type": "array"},  # Metadata for each chunk
                    "total_chunks": {"type": "integer"},  # Total number of chunks
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "extract_metadata",
            self.extract_metadata,
            "Extract metadata from a document",
            {
                "document": {
                    "type": "string",
                    "description": "Document text to extract metadata from",
                },
                "document_type": {
                    "type": "string",
                    "description": "Type of document (e.g., 'financial_report', 'news_article', 'research_paper')",
                    "required": False,
                },
                "metadata_fields": {
                    "type": "array",
                    "description": "List of metadata fields to extract (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "metadata": {"type": "object"},  # Extracted metadata
                    "document_type": {"type": "string"},  # Detected document type
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "clean_document",
            self.clean_document,
            "Clean and normalize document text",
            {
                "document": {
                    "type": "string",
                    "description": "Document text to be cleaned",
                },
                "remove_html": {
                    "type": "boolean",
                    "description": "Whether to remove HTML tags",
                    "default": True,
                },
                "normalize_whitespace": {
                    "type": "boolean",
                    "description": "Whether to normalize whitespace",
                    "default": True,
                },
                "remove_urls": {
                    "type": "boolean",
                    "description": "Whether to remove URLs",
                    "default": False,
                },
                "remove_emails": {
                    "type": "boolean",
                    "description": "Whether to remove email addresses",
                    "default": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "cleaned_document": {"type": "string"},  # Cleaned document text
                    "cleaning_stats": {
                        "type": "object"
                    },  # Statistics about cleaning operations
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "process_document",
            self.process_document,
            "Process a document for embedding generation (clean, extract metadata, and chunk)",
            {
                "document": {
                    "type": "string",
                    "description": "Document text to be processed",
                },
                "document_id": {
                    "type": "string",
                    "description": "Unique identifier for the document (optional)",
                    "required": False,
                },
                "document_type": {
                    "type": "string",
                    "description": "Type of document (optional)",
                    "required": False,
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Size of each chunk in characters",
                    "default": self.default_chunk_size,
                },
                "chunk_overlap": {
                    "type": "integer",
                    "description": "Overlap between consecutive chunks in characters",
                    "default": self.default_chunk_overlap,
                },
                "chunk_strategy": {
                    "type": "string",
                    "description": "Strategy for chunking: 'character', 'word', 'sentence', or 'paragraph'",
                    "default": "paragraph",
                },
            },
            {
                "type": "object",
                "properties": {
                    "chunks": {"type": "array"},  # List of document chunks
                    "chunk_metadata": {"type": "array"},  # Metadata for each chunk
                    "document_metadata": {"type": "object"},  # Document-level metadata
                    "document_id": {"type": "string"},  # Document identifier
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "deduplicate_chunks",
            self.deduplicate_chunks,
            "Remove duplicate or near-duplicate chunks",
            {
                "chunks": {
                    "type": "array",
                    "description": "List of document chunks to deduplicate",
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Threshold for considering chunks as near-duplicates (0.0-1.0)",
                    "default": 0.9,
                },
                "method": {
                    "type": "string",
                    "description": "Method for deduplication: 'exact', 'jaccard', or 'simhash'",
                    "default": "jaccard",
                },
            },
            {
                "type": "object",
                "properties": {
                    "deduplicated_chunks": {"type": "array"},  # Deduplicated chunks
                    "duplicate_count": {
                        "type": "integer"
                    },  # Number of duplicates removed
                    "processing_time": {"type": "number"},
                },
            },
        )

    def chunk_document(
        self,
        document: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = "paragraph",
    ) -> Dict[str, Any]:
        """
        Split a document into chunks for embedding generation.

        Args:
            document: Document text to be chunked
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            chunk_strategy: Strategy for chunking ('character', 'word', 'sentence', or 'paragraph')

        Returns:
            Dictionary with chunked document
        """
        start_time = time.time()

        try:
            # Set default values if not provided
            chunk_size = chunk_size or self.default_chunk_size
            chunk_overlap = chunk_overlap or self.default_chunk_overlap

            # Validate inputs
            if chunk_size <= 0:
                return {
                    "error": "Chunk size must be positive",
                    "processing_time": time.time() - start_time,
                }

            if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                return {
                    "error": "Chunk overlap must be non-negative and less than chunk size",
                    "processing_time": time.time() - start_time,
                }

            # Initialize result
            chunks = []
            chunk_metadata = []

            # Implement chunking based on strategy
            if chunk_strategy == "character":
                # Simple character-based chunking
                for i in range(0, len(document), chunk_size - chunk_overlap):
                    chunk = document[i : i + chunk_size]
                    if chunk:  # Skip empty chunks
                        chunks.append(chunk)
                        chunk_metadata.append(
                            {
                                "start_char": i,
                                "end_char": min(i + chunk_size, len(document)),
                                "chunk_index": len(chunks) - 1,
                            }
                        )

            elif chunk_strategy == "word":
                # Word-based chunking
                words = document.split()
                current_chunk = []
                current_size = 0
                start_char = 0
                char_position = 0

                for i, word in enumerate(words):
                    word_size = len(word) + (
                        1 if i < len(words) - 1 else 0
                    )  # +1 for space, except last word

                    if current_size + word_size > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append(chunk_text)

                        # Calculate end character position
                        end_char = start_char + len(chunk_text)

                        chunk_metadata.append(
                            {
                                "start_char": start_char,
                                "end_char": end_char,
                                "chunk_index": len(chunks) - 1,
                                "word_count": len(current_chunk),
                            }
                        )

                        # Start new chunk with overlap
                        overlap_words = []
                        overlap_size = 0
                        for w in reversed(current_chunk):
                            w_size = len(w) + 1
                            if overlap_size + w_size <= chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_size += w_size
                            else:
                                break
                        current_chunk = overlap_words
                        current_size = sum(len(w) + 1 for w in current_chunk) - (
                            1 if current_chunk else 0
                        )  # Adjust space for last word
                        start_char = (
                            end_char - len(" ".join(current_chunk))
                            if current_chunk
                            else end_char
                        )

                    current_chunk.append(word)
                    current_size += word_size
                    char_position += word_size

                # Add the last chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    chunk_metadata.append(
                        {
                            "start_char": start_char,
                            "end_char": start_char + len(chunk_text),
                            "chunk_index": len(chunks) - 1,
                            "word_count": len(current_chunk),
                        }
                    )

            elif chunk_strategy == "sentence":
                # Sentence-based chunking
                sentences = re.split(r"(?<=[.!?])\s+", document)
                current_chunk = []
                current_size = 0
                start_char = 0
                char_position = 0

                for sentence in sentences:
                    sentence_size = len(sentence) + 1  # +1 for space

                    if current_size + sentence_size > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append(chunk_text)

                        # Calculate end character position
                        end_char = start_char + len(chunk_text)

                        chunk_metadata.append(
                            {
                                "start_char": start_char,
                                "end_char": end_char,
                                "chunk_index": len(chunks) - 1,
                                "sentence_count": len(current_chunk),
                            }
                        )

                        # Start new chunk with overlap
                        overlap_size = 0
                        overlap_sentences = []
                        for s in reversed(current_chunk):
                            s_size = len(s) + 1
                            if overlap_size + s_size <= chunk_overlap:
                                overlap_sentences.insert(0, s)
                                overlap_size += s_size
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_size = sum(len(s) + 1 for s in current_chunk) - (
                            1 if current_chunk else 0
                        )
                        start_char = (
                            end_char - len(" ".join(current_chunk))
                            if current_chunk
                            else end_char
                        )

                    current_chunk.append(sentence)
                    current_size += sentence_size
                    char_position += sentence_size

                # Add the last chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    chunk_metadata.append(
                        {
                            "start_char": start_char,
                            "end_char": start_char + len(chunk_text),
                            "chunk_index": len(chunks) - 1,
                            "sentence_count": len(current_chunk),
                        }
                    )

            elif chunk_strategy == "paragraph":
                # Paragraph-based chunking
                paragraphs = re.split(r"\n\s*\n", document)
                current_chunk = []
                current_size = 0
                start_char = 0
                char_position = 0

                for paragraph in paragraphs:
                    # Skip empty paragraphs
                    if not paragraph.strip():
                        char_position += len(paragraph) + 2  # +2 for newlines
                        continue

                    paragraph_size = len(paragraph) + 2  # +2 for newlines

                    if current_size + paragraph_size > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(chunk_text)

                        # Calculate end character position
                        end_char = start_char + len(chunk_text)

                        chunk_metadata.append(
                            {
                                "start_char": start_char,
                                "end_char": end_char,
                                "chunk_index": len(chunks) - 1,
                                "paragraph_count": len(current_chunk),
                            }
                        )

                        # Start new chunk with overlap
                        overlap_size = 0
                        overlap_paragraphs = []
                        for p in reversed(current_chunk):
                            p_size = len(p) + 2
                            if overlap_size + p_size <= chunk_overlap:
                                overlap_paragraphs.insert(0, p)
                                overlap_size += p_size
                            else:
                                break
                        current_chunk = overlap_paragraphs
                        current_size = sum(len(p) + 2 for p in current_chunk) - (
                            2 if current_chunk else 0
                        )
                        start_char = (
                            end_char - len("\n\n".join(current_chunk))
                            if current_chunk
                            else end_char
                        )

                    current_chunk.append(paragraph)
                    current_size += paragraph_size
                    char_position += paragraph_size

                # Add the last chunk if not empty
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(chunk_text)
                    chunk_metadata.append(
                        {
                            "start_char": start_char,
                            "end_char": start_char + len(chunk_text),
                            "chunk_index": len(chunks) - 1,
                            "paragraph_count": len(current_chunk),
                        }
                    )
            else:
                return {
                    "error": f"Unsupported chunking strategy: {chunk_strategy}",
                    "processing_time": time.time() - start_time,
                }

            # Add document hash to each chunk metadata
            document_hash = hashlib.md5(document.encode()).hexdigest()
            for metadata in chunk_metadata:
                metadata["document_hash"] = document_hash
                metadata["chunk_strategy"] = chunk_strategy
                metadata["timestamp"] = datetime.now().isoformat()

            return {
                "chunks": chunks,
                "chunk_metadata": chunk_metadata,
                "total_chunks": len(chunks),
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error chunking document: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def extract_metadata(
        self,
        document: str,
        document_type: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from a document.

        Args:
            document: Document text to extract metadata from
            document_type: Type of document
            metadata_fields: List of metadata fields to extract

        Returns:
            Dictionary with extracted metadata
        """
        start_time = time.time()

        try:
            # Initialize metadata dictionary
            metadata = {}

            # Add basic metadata
            metadata["character_count"] = len(document)
            metadata["word_count"] = len(document.split())
            metadata["line_count"] = document.count("\n") + 1

            # Extract document hash
            metadata["document_hash"] = hashlib.md5(document.encode()).hexdigest()

            # Detect document type if not provided
            detected_document_type = document_type
            if not detected_document_type:
                # Simple heuristic document type detection
                if re.search(
                    r"(quarterly|annual|financial|report|balance sheet|income statement|cash flow)",
                    document.lower(),
                ):
                    detected_document_type = "financial_report"
                elif re.search(
                    r"(news|article|published|reported|announced)", document.lower()
                ):
                    detected_document_type = "news_article"
                elif re.search(
                    r"(research|study|findings|methodology|conclusion|abstract)",
                    document.lower(),
                ):
                    detected_document_type = "research_paper"
                elif re.search(
                    r"(contract|agreement|terms|parties|clause|hereby)",
                    document.lower(),
                ):
                    detected_document_type = "legal_document"
                else:
                    detected_document_type = "general"

            metadata["document_type"] = detected_document_type

            # Extract title (first non-empty line or first sentence)
            lines = document.strip().split("\n")
            title = next((line for line in lines if line.strip()), "")
            if len(title) > 100:  # If first line is too long, use first sentence
                first_sentence_match = re.search(r"^([^.!?]+[.!?])", document.strip())
                if first_sentence_match:
                    title = first_sentence_match.group(1)

            metadata["title"] = title[:200] if title else ""  # Limit title length

            # Extract date mentions
            date_patterns = [
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY or DD/MM/YYYY
                r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",  # YYYY/MM/DD
                r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",  # Month DD, YYYY
                r"\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b",  # DD Month YYYY
            ]

            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, document, re.IGNORECASE))

            metadata["dates"] = dates[:10]  # Limit to first 10 dates

            # Extract specific fields based on document type
            if detected_document_type == "financial_report":
                # Look for financial metrics
                revenue_match = re.search(
                    r"revenue[s]?[\s:]*[$€£]?[\s]*(\d[\d,.]*)", document.lower()
                )
                if revenue_match:
                    metadata["revenue"] = revenue_match.group(1)

                profit_match = re.search(
                    r"(net income|profit|earnings)[\s:]*[$€£]?[\s]*(\d[\d,.]*)",
                    document.lower(),
                )
                if profit_match:
                    metadata["profit"] = profit_match.group(2)

                # Extract year/quarter references
                year_match = re.search(
                    r"(fiscal|financial)?\s*year\s*(\d{4})", document.lower()
                )
                if year_match:
                    metadata["fiscal_year"] = year_match.group(2)

                quarter_match = re.search(r"q[1-4]\s*(\d{4})", document.lower())
                if quarter_match:
                    metadata["quarter"] = quarter_match.group(0)

            elif detected_document_type == "news_article":
                # Look for author
                author_patterns = [
                    r"by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"author[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"written by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                ]

                for pattern in author_patterns:
                    author_match = re.search(pattern, document)
                    if author_match:
                        metadata["author"] = author_match.group(1)
                        break

                # Look for publication date
                pub_date_match = re.search(
                    r"published on[:\s]+(.{5,25}?)[\n\.]", document.lower()
                )
                if pub_date_match:
                    metadata["publication_date"] = pub_date_match.group(1).strip()

            elif detected_document_type == "research_paper":
                # Extract abstract
                abstract_match = re.search(
                    r"abstract[:\s]+(.*?)(?=\n\s*\n|introduction)",
                    document.lower(),
                    re.DOTALL,
                )
                if abstract_match:
                    metadata["abstract"] = abstract_match.group(1).strip()[
                        :500
                    ]  # Limit length

                # Extract authors
                authors_match = re.search(
                    r"authors?[:\s]+(.*?)(?=\n\s*\n|abstract)",
                    document.lower(),
                    re.DOTALL,
                )
                if authors_match:
                    metadata["authors"] = authors_match.group(1).strip()

            # Extract custom metadata fields if specified
            if metadata_fields:
                for field in metadata_fields:
                    if field not in metadata:
                        # Try to find the field in the document
                        field_pattern = re.compile(
                            f"{field}[:\s]+(.+?)(?=\n\s*\n|\n[A-Z])",
                            re.IGNORECASE | re.DOTALL,
                        )
                        field_match = field_pattern.search(document)
                        if field_match:
                            metadata[field] = field_match.group(1).strip()[
                                :500
                            ]  # Limit length

            return {
                "metadata": metadata,
                "document_type": detected_document_type,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def clean_document(
        self,
        document: str,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
    ) -> Dict[str, Any]:
        """
        Clean and normalize document text.

        Args:
            document: Document text to be cleaned
            remove_html: Whether to remove HTML tags
            normalize_whitespace: Whether to normalize whitespace
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses

        Returns:
            Dictionary with cleaned document
        """
        start_time = time.time()

        try:
            # Initialize cleaning statistics
            cleaning_stats = {
                "original_length": len(document),
                "html_tags_removed": 0,
                "urls_removed": 0,
                "emails_removed": 0,
                "whitespace_normalized": False,
            }

            cleaned_document = document

            # Remove HTML tags
            if remove_html:
                html_pattern = re.compile(r"<[^>]+>")
                html_tags = html_pattern.findall(cleaned_document)
                cleaning_stats["html_tags_removed"] = len(html_tags)
                cleaned_document = html_pattern.sub(" ", cleaned_document)

            # Remove URLs
            if remove_urls:
                url_pattern = re.compile(r"https?://\S+|www\.\S+")
                urls = url_pattern.findall(cleaned_document)
                cleaning_stats["urls_removed"] = len(urls)
                cleaned_document = url_pattern.sub(" ", cleaned_document)

            # Remove email addresses
            if remove_emails:
                email_pattern = re.compile(r"\S+@\S+\.\S+")
                emails = email_pattern.findall(cleaned_document)
                cleaning_stats["emails_removed"] = len(emails)
                cleaned_document = email_pattern.sub(" ", cleaned_document)

            # Normalize whitespace
            if normalize_whitespace:
                # Replace multiple spaces with a single space
                cleaned_document = re.sub(r"\s+", " ", cleaned_document)
                # Replace multiple newlines with a single newline
                cleaned_document = re.sub(r"\n+", "\n", cleaned_document)
                # Remove leading/trailing whitespace
                cleaned_document = cleaned_document.strip()
                cleaning_stats["whitespace_normalized"] = True

            # Update cleaning statistics
            cleaning_stats["cleaned_length"] = len(cleaned_document)
            cleaning_stats["reduction_percentage"] = round(
                (1 - len(cleaned_document) / max(1, len(document))) * 100, 2
            )

            return {
                "cleaned_document": cleaned_document,
                "cleaning_stats": cleaning_stats,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error cleaning document: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def process_document(
        self,
        document: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = "paragraph",
    ) -> Dict[str, Any]:
        """
        Process a document for embedding generation (clean, extract metadata, and chunk).

        Args:
            document: Document text to be processed
            document_id: Unique identifier for the document
            document_type: Type of document
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            chunk_strategy: Strategy for chunking

        Returns:
            Dictionary with processed document
        """
        start_time = time.time()

        try:
            # Generate document ID if not provided
            if not document_id:
                document_id = hashlib.md5(document.encode()).hexdigest()

            # Step 1: Clean the document
            clean_result = self.clean_document(
                document,
                remove_html=True,
                normalize_whitespace=True,
                remove_urls=False,
                remove_emails=False,
            )

            if "error" in clean_result:
                return {
                    "error": f"Error cleaning document: {clean_result['error']}",
                    "processing_time": time.time() - start_time,
                }

            cleaned_document = clean_result["cleaned_document"]

            # Step 2: Extract metadata
            metadata_result = self.extract_metadata(
                cleaned_document, document_type=document_type
            )

            if "error" in metadata_result:
                return {
                    "error": f"Error extracting metadata: {metadata_result['error']}",
                    "processing_time": time.time() - start_time,
                }

            document_metadata = metadata_result["metadata"]
            detected_document_type = metadata_result["document_type"]

            # Add document ID to metadata
            document_metadata["document_id"] = document_id

            # Step 3: Chunk the document
            chunk_result = self.chunk_document(
                cleaned_document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_strategy=chunk_strategy,
            )

            if "error" in chunk_result:
                return {
                    "error": f"Error chunking document: {chunk_result['error']}",
                    "processing_time": time.time() - start_time,
                }

            chunks = chunk_result["chunks"]
            chunk_metadata = chunk_result["chunk_metadata"]

            # Add document metadata to each chunk metadata
            for metadata in chunk_metadata:
                metadata["document_id"] = document_id
                metadata["document_type"] = detected_document_type

                # Add title to chunk metadata if available
                if "title" in document_metadata:
                    metadata["document_title"] = document_metadata["title"]

            return {
                "chunks": chunks,
                "chunk_metadata": chunk_metadata,
                "document_metadata": document_metadata,
                "document_id": document_id,
                "document_type": detected_document_type,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def deduplicate_chunks(
        self,
        chunks: List[str],
        similarity_threshold: float = 0.9,
        method: str = "jaccard",
    ) -> Dict[str, Any]:
        """
        Remove duplicate or near-duplicate chunks.

        Args:
            chunks: List of document chunks to deduplicate
            similarity_threshold: Threshold for considering chunks as near-duplicates
            method: Method for deduplication ('exact', 'jaccard', 'simhash')

        Returns:
            Dictionary with deduplicated chunks
        """
        start_time = time.time()

        try:
            if not chunks:
                return {
                    "deduplicated_chunks": [],
                    "duplicate_count": 0,
                    "processing_time": time.time() - start_time,
                }

            deduplicated_chunks = []
            seen_hashes = set()
            duplicate_count = 0

            if method == "exact":
                for chunk in chunks:
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                    if chunk_hash not in seen_hashes:
                        deduplicated_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                    else:
                        duplicate_count += 1

            elif method == "jaccard":
                # Simple Jaccard similarity implementation
                def jaccard_similarity(set1, set2):
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    return intersection / union if union > 0 else 0

                chunk_sets = [set(chunk.split()) for chunk in chunks]
                keep_indices = list(range(len(chunks)))

                for i in range(len(chunks)):
                    if i not in keep_indices:
                        continue
                    for j in range(i + 1, len(chunks)):
                        if j not in keep_indices:
                            continue
                        similarity = jaccard_similarity(chunk_sets[i], chunk_sets[j])
                        if similarity >= similarity_threshold:
                            keep_indices.remove(j)
                            duplicate_count += 1

                deduplicated_chunks = [chunks[i] for i in keep_indices]

            elif method == "simhash":
                # Placeholder for Simhash implementation
                # Requires a library like 'simhash'
                self.logger.warning(
                    "Simhash deduplication method is not implemented yet. Using exact matching."
                )
                for chunk in chunks:
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                    if chunk_hash not in seen_hashes:
                        deduplicated_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                    else:
                        duplicate_count += 1
            else:
                return {
                    "error": f"Unsupported deduplication method: {method}",
                    "processing_time": time.time() - start_time,
                }

            return {
                "deduplicated_chunks": deduplicated_chunks,
                "duplicate_count": duplicate_count,
                "original_count": len(chunks),
                "method": method,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error deduplicating chunks: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
