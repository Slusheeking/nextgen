"""
Relevance Feedback MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
relevance feedback capabilities for improving retrieval results.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-relevance-feedback",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class RelevanceFeedbackMCP(BaseMCPServer):
    """
    MCP server for incorporating feedback into retrieval.

    This tool implements explicit and implicit feedback handling to improve
    retrieval results based on user interactions and feedback.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Relevance Feedback MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - feedback_store: Storage configuration for feedback data
                - feedback_weight: Weight to apply to feedback (default: 0.5)
                - cache_dir: Directory for caching intermediate results
                - min_feedback_count: Minimum feedback count for reliable reranking
                - feedback_decay_factor: Time decay factor for older feedback
        """
        super().__init__(name="relevance_feedback_mcp", config=config)

        # Set default configurations
        self.feedback_store = self.config.get(
            "feedback_store",
            {
                "type": "memory",  # Options: "memory", "redis", "file"
                "path": "./feedback_data",  # For file storage
                "redis_key_prefix": "feedback:",  # For Redis storage
            },
        )
        self.feedback_weight = self.config.get("feedback_weight", 0.5)
        self.cache_dir = self.config.get("cache_dir", "./feedback_cache")
        self.min_feedback_count = self.config.get("min_feedback_count", 3)
        self.feedback_decay_factor = self.config.get(
            "feedback_decay_factor", 0.9
        )  # 0.9 means 10% decay per time unit

        # Initialize feedback storage
        self._init_feedback_store()

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _init_feedback_store(self):
        """Initialize the feedback storage system."""
        store_type = self.feedback_store.get("type", "memory")

        if store_type == "memory":
            # Simple in-memory storage
            self.explicit_feedback = {}  # {query_id: {doc_id: {rating: float, timestamp: str}}}
            self.implicit_feedback = {}  # {query_id: {doc_id: {clicks: int, dwell_time: float, timestamp: str}}}
            self.query_doc_pairs = {}  # {query_id: {doc_id: {query: str, doc_text: str}}}
            self.logger.info("Initialized in-memory feedback store")

        elif store_type == "file":
            # File-based storage
            path = self.feedback_store.get("path", "./feedback_data")
            os.makedirs(path, exist_ok=True)

            # Load existing data if available
            explicit_path = os.path.join(path, "explicit_feedback.json")
            implicit_path = os.path.join(path, "implicit_feedback.json")
            pairs_path = os.path.join(path, "query_doc_pairs.json")

            try:
                if os.path.exists(explicit_path):
                    with open(explicit_path, "r") as f:
                        self.explicit_feedback = json.load(f)
                else:
                    self.explicit_feedback = {}

                if os.path.exists(implicit_path):
                    with open(implicit_path, "r") as f:
                        self.implicit_feedback = json.load(f)
                else:
                    self.implicit_feedback = {}

                if os.path.exists(pairs_path):
                    with open(pairs_path, "r") as f:
                        self.query_doc_pairs = json.load(f)
                else:
                    self.query_doc_pairs = {}

                self.logger.info(f"Loaded feedback data from {path}")
            except Exception as e:
                self.logger.error(f"Error loading feedback data: {e}")
                self.explicit_feedback = {}
                self.implicit_feedback = {}
                self.query_doc_pairs = {}

        elif store_type == "redis":
            # Redis-based storage (placeholder - would use RedisMCP in real implementation)
            self.redis_key_prefix = self.feedback_store.get(
                "redis_key_prefix", "feedback:"
            )
            # In a real implementation, we would initialize Redis client here
            # For now, we'll use in-memory as a fallback
            self.explicit_feedback = {}
            self.implicit_feedback = {}
            self.query_doc_pairs = {}
            self.logger.warning(
                "Redis storage not fully implemented, using in-memory fallback"
            )

        else:
            self.logger.error(f"Unsupported feedback store type: {store_type}")
            # Fallback to in-memory
            self.explicit_feedback = {}
            self.implicit_feedback = {}
            self.query_doc_pairs = {}

    def _save_feedback_data(self):
        """Save feedback data if using file storage."""
        store_type = self.feedback_store.get("type", "memory")

        if store_type == "file":
            path = self.feedback_store.get("path", "./feedback_data")

            try:
                explicit_path = os.path.join(path, "explicit_feedback.json")
                with open(explicit_path, "w") as f:
                    json.dump(self.explicit_feedback, f)

                implicit_path = os.path.join(path, "implicit_feedback.json")
                with open(implicit_path, "w") as f:
                    json.dump(self.implicit_feedback, f)

                pairs_path = os.path.join(path, "query_doc_pairs.json")
                with open(pairs_path, "w") as f:
                    json.dump(self.query_doc_pairs, f)

                self.logger.info(f"Saved feedback data to {path}")
            except Exception as e:
                self.logger.error(f"Error saving feedback data: {e}")

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "record_explicit_feedback",
            self.record_explicit_feedback,
            "Record explicit user feedback on document relevance",
            {
                "query_id": {
                    "type": "string",
                    "description": "Identifier for the query",
                },
                "document_id": {
                    "type": "string",
                    "description": "Identifier for the document",
                },
                "rating": {
                    "type": "number",
                    "description": "Relevance rating (0.0-1.0, where 1.0 is most relevant)",
                },
                "query_text": {
                    "type": "string",
                    "description": "Original query text (optional)",
                    "required": False,
                },
                "document_text": {
                    "type": "string",
                    "description": "Document text or snippet (optional)",
                    "required": False,
                },
                "user_id": {
                    "type": "string",
                    "description": "Identifier for the user providing feedback (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean"
                    },  # Whether feedback was recorded successfully
                    "feedback_id": {
                        "type": "string"
                    },  # Identifier for the feedback record
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "record_implicit_feedback",
            self.record_implicit_feedback,
            "Record implicit user feedback based on interactions",
            {
                "query_id": {
                    "type": "string",
                    "description": "Identifier for the query",
                },
                "document_id": {
                    "type": "string",
                    "description": "Identifier for the document",
                },
                "interaction_type": {
                    "type": "string",
                    "description": "Type of interaction (e.g., 'click', 'dwell_time', 'bookmark')",
                },
                "interaction_value": {
                    "type": "number",
                    "description": "Value associated with the interaction (e.g., dwell time in seconds)",
                },
                "query_text": {
                    "type": "string",
                    "description": "Original query text (optional)",
                    "required": False,
                },
                "document_text": {
                    "type": "string",
                    "description": "Document text or snippet (optional)",
                    "required": False,
                },
                "user_id": {
                    "type": "string",
                    "description": "Identifier for the user providing feedback (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean"
                    },  # Whether feedback was recorded successfully
                    "feedback_id": {
                        "type": "string"
                    },  # Identifier for the feedback record
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "rerank_results",
            self.rerank_results,
            "Rerank search results based on feedback",
            {
                "query_id": {
                    "type": "string",
                    "description": "Identifier for the query",
                },
                "results": {
                    "type": "array",
                    "description": "List of search results to rerank",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "score": {"type": "number"},
                            "text": {"type": "string", "required": False},
                        },
                    },
                },
                "use_explicit_feedback": {
                    "type": "boolean",
                    "description": "Whether to use explicit feedback for reranking",
                    "default": True,
                },
                "use_implicit_feedback": {
                    "type": "boolean",
                    "description": "Whether to use implicit feedback for reranking",
                    "default": True,
                },
                "feedback_weight": {
                    "type": "number",
                    "description": "Weight to apply to feedback scores (0.0-1.0)",
                    "default": 0.5,
                },
            },
            {
                "type": "object",
                "properties": {
                    "reranked_results": {"type": "array"},  # Reranked results
                    "feedback_applied": {
                        "type": "boolean"
                    },  # Whether feedback was applied
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "get_feedback_stats",
            self.get_feedback_stats,
            "Get statistics about recorded feedback",
            {
                "query_id": {
                    "type": "string",
                    "description": "Identifier for the query (optional)",
                    "required": False,
                },
                "document_id": {
                    "type": "string",
                    "description": "Identifier for the document (optional)",
                    "required": False,
                },
                "user_id": {
                    "type": "string",
                    "description": "Identifier for the user (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "explicit_feedback_count": {
                        "type": "integer"
                    },  # Count of explicit feedback records
                    "implicit_feedback_count": {
                        "type": "integer"
                    },  # Count of implicit feedback records
                    "average_ratings": {
                        "type": "object"
                    },  # Average ratings by query/document
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "generate_query_suggestions",
            self.generate_query_suggestions,
            "Generate query suggestions based on feedback",
            {
                "query_id": {
                    "type": "string",
                    "description": "Identifier for the original query",
                },
                "query_text": {"type": "string", "description": "Original query text"},
                "max_suggestions": {
                    "type": "integer",
                    "description": "Maximum number of suggestions to generate",
                    "default": 3,
                },
            },
            {
                "type": "object",
                "properties": {
                    "suggestions": {"type": "array"},  # List of query suggestions
                    "original_query": {"type": "string"},  # Original query
                    "processing_time": {"type": "number"},
                },
            },
        )

    def record_explicit_feedback(
        self,
        query_id: str,
        document_id: str,
        rating: float,
        query_text: Optional[str] = None,
        document_text: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record explicit user feedback on document relevance.

        Args:
            query_id: Identifier for the query
            document_id: Identifier for the document
            rating: Relevance rating (0.0-1.0, where 1.0 is most relevant)
            query_text: Original query text (optional)
            document_text: Document text or snippet (optional)
            user_id: Identifier for the user providing feedback (optional)

        Returns:
            Dictionary with feedback recording status
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not query_id or not document_id:
                return {
                    "success": False,
                    "error": "Query ID and Document ID are required",
                    "processing_time": time.time() - start_time,
                }

            if rating < 0.0 or rating > 1.0:
                return {
                    "success": False,
                    "error": "Rating must be between 0.0 and 1.0",
                    "processing_time": time.time() - start_time,
                }

            # Generate a feedback ID
            timestamp = datetime.now().isoformat()
            feedback_id = f"explicit_{query_id}_{document_id}_{int(time.time())}"

            # Store the feedback
            if query_id not in self.explicit_feedback:
                self.explicit_feedback[query_id] = {}

            self.explicit_feedback[query_id][document_id] = {
                "rating": rating,
                "timestamp": timestamp,
                "user_id": user_id,
            }

            # Store query and document text if provided
            if query_text or document_text:
                if query_id not in self.query_doc_pairs:
                    self.query_doc_pairs[query_id] = {}

                if document_id not in self.query_doc_pairs[query_id]:
                    self.query_doc_pairs[query_id][document_id] = {}

                if query_text:
                    self.query_doc_pairs[query_id][document_id]["query"] = query_text

                if document_text:
                    self.query_doc_pairs[query_id][document_id]["doc_text"] = (
                        document_text
                    )

            # Save feedback data if using file storage
            self._save_feedback_data()

            # Log the feedback
            self.logger.info(
                f"Recorded explicit feedback: query={query_id}, doc={document_id}, rating={rating}"
            )

            return {
                "success": True,
                "feedback_id": feedback_id,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error recording explicit feedback: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def record_implicit_feedback(
        self,
        query_id: str,
        document_id: str,
        interaction_type: str,
        interaction_value: float,
        query_text: Optional[str] = None,
        document_text: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record implicit user feedback based on interactions.

        Args:
            query_id: Identifier for the query
            document_id: Identifier for the document
            interaction_type: Type of interaction (e.g., 'click', 'dwell_time', 'bookmark')
            interaction_value: Value associated with the interaction
            query_text: Original query text (optional)
            document_text: Document text or snippet (optional)
            user_id: Identifier for the user providing feedback (optional)

        Returns:
            Dictionary with feedback recording status
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not query_id or not document_id:
                return {
                    "success": False,
                    "error": "Query ID and Document ID are required",
                    "processing_time": time.time() - start_time,
                }

            if not interaction_type:
                return {
                    "success": False,
                    "error": "Interaction type is required",
                    "processing_time": time.time() - start_time,
                }

            # Generate a feedback ID
            timestamp = datetime.now().isoformat()
            feedback_id = f"implicit_{query_id}_{document_id}_{interaction_type}_{int(time.time())}"

            # Store the feedback
            if query_id not in self.implicit_feedback:
                self.implicit_feedback[query_id] = {}

            if document_id not in self.implicit_feedback[query_id]:
                self.implicit_feedback[query_id][document_id] = {}

            # Update or create the interaction record
            if interaction_type not in self.implicit_feedback[query_id][document_id]:
                self.implicit_feedback[query_id][document_id][interaction_type] = []

            self.implicit_feedback[query_id][document_id][interaction_type].append(
                {"value": interaction_value, "timestamp": timestamp, "user_id": user_id}
            )

            # Store query and document text if provided
            if query_text or document_text:
                if query_id not in self.query_doc_pairs:
                    self.query_doc_pairs[query_id] = {}

                if document_id not in self.query_doc_pairs[query_id]:
                    self.query_doc_pairs[query_id][document_id] = {}

                if query_text:
                    self.query_doc_pairs[query_id][document_id]["query"] = query_text

                if document_text:
                    self.query_doc_pairs[query_id][document_id]["doc_text"] = (
                        document_text
                    )

            # Save feedback data if using file storage
            self._save_feedback_data()

            # Log the feedback
            self.logger.info(
                f"Recorded implicit feedback: query={query_id}, doc={document_id}, type={interaction_type}, value={interaction_value}"
            )

            return {
                "success": True,
                "feedback_id": feedback_id,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error recording implicit feedback: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _calculate_implicit_score(
        self, implicit_data: Dict[str, Any]
    ) -> Optional[float]:
        """
        Calculate a feedback score from implicit feedback data.

        Args:
            implicit_data: Dictionary containing implicit feedback data

        Returns:
            Calculated score between 0.0 and 1.0, or None if no score could be calculated
        """
        if not implicit_data:
            return None

        score_components = []

        # Process click data
        if "click" in implicit_data:
            clicks = implicit_data["click"]
            if clicks:
                # More clicks generally indicate more relevance
                # Apply time decay to older clicks
                click_score = 0
                for click in clicks:
                    click_value = click.get("value", 1)

                    # Apply time decay if timestamp is available
                    if "timestamp" in click:
                        timestamp = datetime.fromisoformat(click["timestamp"])
                        days_old = (datetime.now() - timestamp).days
                        time_decay = self.feedback_decay_factor**days_old
                        click_value *= time_decay

                    click_score += click_value

                # Normalize click score (diminishing returns after 3 clicks)
                normalized_click_score = min(click_score / 3, 1.0)
                score_components.append(normalized_click_score)

        # Process dwell time data
        if "dwell_time" in implicit_data:
            dwell_times = implicit_data["dwell_time"]
            if dwell_times:
                # Longer dwell times generally indicate more relevance
                # Apply time decay to older dwell times
                dwell_values = []
                for dwell in dwell_times:
                    dwell_value = dwell.get("value", 0)

                    # Apply time decay if timestamp is available
                    if "timestamp" in dwell:
                        timestamp = datetime.fromisoformat(dwell["timestamp"])
                        days_old = (datetime.now() - timestamp).days
                        time_decay = self.feedback_decay_factor**days_old
                        dwell_value *= time_decay

                    dwell_values.append(dwell_value)

                # Use the maximum dwell time as the score component
                # Normalize dwell time (diminishing returns after 60 seconds)
                if dwell_values:
                    max_dwell = max(dwell_values)
                    normalized_dwell_score = min(max_dwell / 60.0, 1.0)
                    score_components.append(normalized_dwell_score)

        # Process bookmark data
        if "bookmark" in implicit_data:
            bookmarks = implicit_data["bookmark"]
            if bookmarks:
                # Bookmarking indicates high relevance
                # Apply time decay to older bookmarks
                bookmark_score = 0
                for bookmark in bookmarks:
                    bookmark_value = bookmark.get("value", 1)

                    # Apply time decay if timestamp is available
                    if "timestamp" in bookmark:
                        timestamp = datetime.fromisoformat(bookmark["timestamp"])
                        days_old = (datetime.now() - timestamp).days
                        time_decay = self.feedback_decay_factor**days_old
                        bookmark_value *= time_decay

                    bookmark_score += bookmark_value

                # Normalize bookmark score (bookmarking is a strong signal)
                normalized_bookmark_score = min(bookmark_score, 1.0)
                score_components.append(
                    normalized_bookmark_score * 0.8
                )  # Bookmarks are weighted highly

        # Calculate final score as weighted average of components
        if score_components:
            return sum(score_components) / len(score_components)

        return None

    def rerank_results(
        self,
        query_id: str,
        results: List[Dict[str, Any]],
        use_explicit_feedback: bool = True,
        use_implicit_feedback: bool = True,
        feedback_weight: float = None,
    ) -> Dict[str, Any]:
        """
        Rerank search results based on feedback.

        Args:
            query_id: Identifier for the query
            results: List of search results to rerank
            use_explicit_feedback: Whether to use explicit feedback for reranking
            use_implicit_feedback: Whether to use implicit feedback for reranking
            feedback_weight: Weight to apply to feedback scores (0.0-1.0)

        Returns:
            Dictionary with reranked results
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not query_id:
                return {
                    "reranked_results": results,
                    "feedback_applied": False,
                    "error": "Query ID is required",
                    "processing_time": time.time() - start_time,
                }

            if not results:
                return {
                    "reranked_results": [],
                    "feedback_applied": False,
                    "processing_time": time.time() - start_time,
                }

            # Set default feedback weight if not provided
            feedback_weight = (
                feedback_weight if feedback_weight is not None else self.feedback_weight
            )

            # Check if we have feedback for this query
            has_explicit_feedback = (
                query_id in self.explicit_feedback and use_explicit_feedback
            )
            has_implicit_feedback = (
                query_id in self.implicit_feedback and use_implicit_feedback
            )

            if not has_explicit_feedback and not has_implicit_feedback:
                return {
                    "reranked_results": results,
                    "feedback_applied": False,
                    "message": "No feedback available for this query",
                    "processing_time": time.time() - start_time,
                }

            # Calculate feedback scores for each result
            reranked_results = []
            feedback_applied = False

            for result in results:
                document_id = result.get("document_id")
                original_score = result.get(
                    "score", 0.5
                )  # Default score if not provided

                # Initialize with original score
                adjusted_score = original_score
                feedback_score = None

                # Apply explicit feedback if available
                if (
                    has_explicit_feedback
                    and document_id in self.explicit_feedback[query_id]
                ):
                    explicit_data = self.explicit_feedback[query_id][document_id]
                    explicit_score = explicit_data.get("rating", 0.5)

                    # Apply time decay if timestamp is available
                    if "timestamp" in explicit_data:
                        timestamp = datetime.fromisoformat(explicit_data["timestamp"])
                        days_old = (datetime.now() - timestamp).days
                        time_decay = self.feedback_decay_factor**days_old
                        explicit_score *= time_decay

                    feedback_score = explicit_score
                    feedback_applied = True

                # Apply implicit feedback if available
                if (
                    has_implicit_feedback
                    and document_id in self.implicit_feedback[query_id]
                ):
                    implicit_data = self.implicit_feedback[query_id][document_id]
                    implicit_score = self._calculate_implicit_score(implicit_data)

                    if implicit_score is not None:
                        if feedback_score is None:
                            feedback_score = implicit_score
                        else:
                            # Combine explicit and implicit scores (simple average)
                            feedback_score = (feedback_score + implicit_score) / 2

                        feedback_applied = True

                # Combine original score with feedback score
                if feedback_score is not None:
                    adjusted_score = (
                        1 - feedback_weight
                    ) * original_score + feedback_weight * feedback_score

                # Create reranked result
                reranked_result = result.copy()
                reranked_result["score"] = adjusted_score
                if feedback_score is not None:
                    reranked_result["feedback_score"] = feedback_score

                reranked_results.append(reranked_result)

            # Sort by adjusted score (descending)
            reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return {
                "reranked_results": reranked_results,
                "feedback_applied": feedback_applied,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error reranking results: {e}")
            return {
                "reranked_results": results,
                "feedback_applied": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def get_feedback_stats(
        self,
        query_id: Optional[str] = None,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about recorded feedback.

        Args:
            query_id: Identifier for the query (optional)
            document_id: Identifier for the document (optional)
            user_id: Identifier for the user (optional)

        Returns:
            Dictionary with feedback statistics
        """
        start_time = time.time()

        try:
            # Initialize counters
            explicit_count = 0
            implicit_count = 0
            average_ratings = {}

            # Process explicit feedback
            if query_id:
                # Stats for a specific query
                if query_id in self.explicit_feedback:
                    query_feedback = self.explicit_feedback[query_id]

                    if document_id:
                        # Stats for a specific document
                        if document_id in query_feedback:
                            doc_feedback = query_feedback[document_id]
                            if not user_id or doc_feedback.get("user_id") == user_id:
                                explicit_count = 1
                                average_ratings[document_id] = doc_feedback.get(
                                    "rating", 0
                                )
                    else:
                        # Stats for all documents in the query
                        ratings = []
                        for doc_id, doc_feedback in query_feedback.items():
                            if not user_id or doc_feedback.get("user_id") == user_id:
                                explicit_count += 1
                                ratings.append(doc_feedback.get("rating", 0))
                                average_ratings[doc_id] = doc_feedback.get("rating", 0)

                        if ratings:
                            average_ratings["overall"] = sum(ratings) / len(ratings)
            else:
                # Stats for all queries
                for q_id, query_feedback in self.explicit_feedback.items():
                    query_ratings = []

                    for doc_id, doc_feedback in query_feedback.items():
                        if (not document_id or doc_id == document_id) and (
                            not user_id or doc_feedback.get("user_id") == user_id
                        ):
                            explicit_count += 1
                            query_ratings.append(doc_feedback.get("rating", 0))

                    if query_ratings:
                        average_ratings[q_id] = sum(query_ratings) / len(query_ratings)

            # Process implicit feedback
            if query_id:
                # Stats for a specific query
                if query_id in self.implicit_feedback:
                    query_feedback = self.implicit_feedback[query_id]

                    if document_id:
                        # Stats for a specific document
                        if document_id in query_feedback:
                            doc_feedback = query_feedback[document_id]
                            for interaction_type, interactions in doc_feedback.items():
                                for interaction in interactions:
                                    if (
                                        not user_id
                                        or interaction.get("user_id") == user_id
                                    ):
                                        implicit_count += 1
                    else:
                        # Stats for all documents in the query
                        for doc_id, doc_feedback in query_feedback.items():
                            for interaction_type, interactions in doc_feedback.items():
                                for interaction in interactions:
                                    if (
                                        not user_id
                                        or interaction.get("user_id") == user_id
                                    ):
                                        implicit_count += 1
            else:
                # Stats for all queries
                for q_id, query_feedback in self.implicit_feedback.items():
                    for doc_id, doc_feedback in query_feedback.items():
                        if not document_id or doc_id == document_id:
                            for interaction_type, interactions in doc_feedback.items():
                                for interaction in interactions:
                                    if (
                                        not user_id
                                        or interaction.get("user_id") == user_id
                                    ):
                                        implicit_count += 1

            return {
                "explicit_feedback_count": explicit_count,
                "implicit_feedback_count": implicit_count,
                "average_ratings": average_ratings,
                "query_id": query_id,
                "document_id": document_id,
                "user_id": user_id,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error getting feedback stats: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def generate_query_suggestions(
        self, query_id: str, query_text: str, max_suggestions: int = 3
    ) -> Dict[str, Any]:
        """
        Generate query suggestions based on feedback.

        Args:
            query_id: Identifier for the original query
            query_text: Original query text
            max_suggestions: Maximum number of suggestions to generate

        Returns:
            Dictionary with query suggestions
        """
        start_time = time.time()

        try:
            if not query_id or not query_text:
                return {
                    "suggestions": [],
                    "original_query": query_text,
                    "error": "Query ID and query text are required",
                    "processing_time": time.time() - start_time,
                }

            suggestions = []

            # Check if we have feedback for this query
            has_explicit_feedback = query_id in self.explicit_feedback
            has_implicit_feedback = query_id in self.implicit_feedback

            if not has_explicit_feedback and not has_implicit_feedback:
                # No feedback available, generate generic suggestions
                suggestions = self._generate_generic_suggestions(
                    query_text, max_suggestions
                )
            else:
                # Generate suggestions based on feedback
                suggestions = self._generate_feedback_based_suggestions(
                    query_id, query_text, max_suggestions
                )

            return {
                "suggestions": suggestions,
                "original_query": query_text,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error generating query suggestions: {e}")
            return {
                "suggestions": [],
                "original_query": query_text,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _generate_generic_suggestions(
        self, query_text: str, max_suggestions: int
    ) -> List[str]:
        """Generate generic query suggestions without feedback data."""
        suggestions = []

        # Simple word-based suggestions
        words = query_text.split()

        # Suggestion 1: More specific (add a qualifier)
        if len(words) >= 2:
            qualifiers = ["detailed", "comprehensive", "recent", "popular", "advanced"]
            for qualifier in qualifiers:
                if qualifier not in query_text.lower():
                    suggestions.append(f"{qualifier} {query_text}")
                    break

        # Suggestion 2: Broader (remove last word if more than 3 words)
        if len(words) > 3:
            suggestions.append(" ".join(words[:-1]))

        # Suggestion 3: Alternative phrasing
        if query_text.lower().startswith("how"):
            suggestions.append(f"ways to {query_text[3:].strip()}")
        elif query_text.lower().startswith("what is"):
            suggestions.append(f"definition of {query_text[7:].strip()}")
        elif query_text.lower().startswith("where"):
            suggestions.append(f"location of {query_text[5:].strip()}")

        # Limit to max_suggestions
        return suggestions[:max_suggestions]

    def _generate_feedback_based_suggestions(
        self, query_id: str, query_text: str, max_suggestions: int
    ) -> List[str]:
        """Generate query suggestions based on feedback data."""
        suggestions = []

        # Get highly rated documents for this query
        highly_rated_docs = []

        if query_id in self.explicit_feedback:
            for doc_id, feedback in self.explicit_feedback[query_id].items():
                if feedback.get("rating", 0) >= 0.7:  # Only consider highly rated docs
                    highly_rated_docs.append(doc_id)

        # Extract terms from highly rated documents
        terms = set()
        for doc_id in highly_rated_docs:
            if (
                query_id in self.query_doc_pairs
                and doc_id in self.query_doc_pairs[query_id]
            ):
                doc_text = self.query_doc_pairs[query_id][doc_id].get("doc_text", "")
                if doc_text:
                    # Extract important terms (simple approach)
                    doc_words = set(re.findall(r"\b\w{4,}\b", doc_text.lower()))
                    query_words = set(re.findall(r"\b\w{4,}\b", query_text.lower()))

                    # Find terms in document that aren't in the query
                    new_terms = doc_words - query_words
                    terms.update(new_terms)

        # Generate suggestions using extracted terms
        if terms:
            # Suggestion 1: Add a relevant term
            for term in sorted(terms, key=len, reverse=True)[:3]:
                if term not in query_text.lower():
                    suggestions.append(f"{query_text} {term}")

            # Suggestion 2: Replace a term
            words = query_text.split()
            if len(words) >= 3:
                for i in range(len(words)):
                    for term in sorted(terms)[:2]:
                        if term not in words[i].lower():
                            new_words = words.copy()
                            new_words[i] = term
                            suggestions.append(" ".join(new_words))

        # If we couldn't generate enough suggestions, add some generic ones
        if len(suggestions) < max_suggestions:
            generic_suggestions = self._generate_generic_suggestions(
                query_text, max_suggestions - len(suggestions)
            )
            suggestions.extend(generic_suggestions)

        # Limit to max_suggestions and remove duplicates
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if (
                suggestion.lower() not in seen
                and suggestion.lower() != query_text.lower()
            ):
                unique_suggestions.append(suggestion)
                seen.add(suggestion.lower())
                if len(unique_suggestions) >= max_suggestions:
                    break

        return unique_suggestions
