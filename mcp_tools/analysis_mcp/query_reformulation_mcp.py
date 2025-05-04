"""
Query Reformulation MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
query reformulation capabilities for improving RAG query quality.
"""

import os
import time
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-query-reformulation",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class QueryReformulationMCP(BaseMCPServer):
    """
    MCP server for improving RAG query quality.

    This tool implements query expansion and domain-specific enhancement
    to improve the quality of queries for retrieval augmented generation systems.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Query Reformulation MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - domain_specific_terms: Dictionary mapping domains to relevant terms
                - expansion_strategies: List of enabled expansion strategies
                - cache_dir: Directory for caching intermediate results
                - max_expansion_terms: Maximum number of terms to add in expansion
                - min_term_relevance: Minimum relevance score for expansion terms
        """
        super().__init__(name="query_reformulation_mcp", config=config)

        # Set default configurations
        self.domain_specific_terms = self.config.get(
            "domain_specific_terms",
            {
                "finance": [
                    "stock",
                    "market",
                    "investment",
                    "portfolio",
                    "asset",
                    "equity",
                    "bond",
                    "derivative",
                    "hedge",
                    "risk",
                ],
                "technology": [
                    "software",
                    "hardware",
                    "algorithm",
                    "data",
                    "cloud",
                    "network",
                    "security",
                    "interface",
                    "system",
                    "platform",
                ],
                "healthcare": [
                    "patient",
                    "treatment",
                    "diagnosis",
                    "clinical",
                    "medical",
                    "therapy",
                    "pharmaceutical",
                    "disease",
                    "health",
                    "care",
                ],
            },
        )
        self.expansion_strategies = self.config.get(
            "expansion_strategies", ["synonyms", "domain_terms", "query_decomposition"]
        )
        self.cache_dir = self.config.get("cache_dir", "./query_cache")
        self.max_expansion_terms = self.config.get("max_expansion_terms", 5)
        self.min_term_relevance = self.config.get("min_term_relevance", 0.5)

        # Common synonyms for query expansion
        self.synonym_mappings = self.config.get(
            "synonym_mappings",
            {
                "increase": ["rise", "growth", "gain", "appreciation", "uptrend"],
                "decrease": ["decline", "drop", "fall", "reduction", "downtrend"],
                "analyze": ["examine", "evaluate", "assess", "study", "investigate"],
                "predict": ["forecast", "project", "estimate", "anticipate", "expect"],
                "impact": ["effect", "influence", "consequence", "result", "outcome"],
                "strategy": ["approach", "plan", "method", "technique", "framework"],
                "risk": ["hazard", "danger", "exposure", "vulnerability", "threat"],
                "performance": [
                    "result",
                    "achievement",
                    "accomplishment",
                    "execution",
                    "output",
                ],
                "trend": ["pattern", "direction", "movement", "tendency", "drift"],
                "correlation": [
                    "relationship",
                    "association",
                    "connection",
                    "link",
                    "interdependence",
                ],
            },
        )

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "expand_query",
            self.expand_query,
            "Expand a query with related terms to improve retrieval",
            {
                "query": {"type": "string", "description": "Original query to expand"},
                "domain": {
                    "type": "string",
                    "description": "Domain context for query expansion (e.g., 'finance', 'technology')",
                    "required": False,
                },
                "strategies": {
                    "type": "array",
                    "description": "Expansion strategies to use (e.g., 'synonyms', 'domain_terms', 'query_decomposition')",
                    "required": False,
                },
                "max_terms": {
                    "type": "integer",
                    "description": "Maximum number of terms to add in expansion",
                    "default": self.max_expansion_terms,
                },
            },
            {
                "type": "object",
                "properties": {
                    "expanded_query": {"type": "string"},  # Expanded query
                    "expansion_terms": {"type": "array"},  # Terms added to the query
                    "strategies_applied": {
                        "type": "array"
                    },  # Strategies that were applied
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "decompose_query",
            self.decompose_query,
            "Break down a complex query into simpler sub-queries",
            {
                "query": {
                    "type": "string",
                    "description": "Complex query to decompose",
                },
                "max_sub_queries": {
                    "type": "integer",
                    "description": "Maximum number of sub-queries to generate",
                    "default": 3,
                },
            },
            {
                "type": "object",
                "properties": {
                    "sub_queries": {"type": "array"},  # List of sub-queries
                    "main_focus": {
                        "type": "string"
                    },  # Main focus of the original query
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "enhance_domain_specificity",
            self.enhance_domain_specificity,
            "Enhance a query with domain-specific terminology",
            {
                "query": {"type": "string", "description": "Original query to enhance"},
                "domain": {
                    "type": "string",
                    "description": "Domain for terminology enhancement (e.g., 'finance', 'technology')",
                },
                "max_terms": {
                    "type": "integer",
                    "description": "Maximum number of domain terms to add",
                    "default": 3,
                },
            },
            {
                "type": "object",
                "properties": {
                    "enhanced_query": {"type": "string"},  # Domain-enhanced query
                    "added_terms": {"type": "array"},  # Domain terms added
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "add_synonym_expansion",
            self.add_synonym_expansion,
            "Expand a query with synonyms of key terms",
            {
                "query": {
                    "type": "string",
                    "description": "Original query to expand with synonyms",
                },
                "max_synonyms_per_term": {
                    "type": "integer",
                    "description": "Maximum number of synonyms to add per term",
                    "default": 2,
                },
            },
            {
                "type": "object",
                "properties": {
                    "expanded_query": {"type": "string"},  # Synonym-expanded query
                    "synonym_mappings": {
                        "type": "object"
                    },  # Terms and their synonyms used
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "optimize_for_vector_search",
            self.optimize_for_vector_search,
            "Optimize a query for vector-based semantic search",
            {
                "query": {
                    "type": "string",
                    "description": "Original query to optimize",
                },
                "remove_stopwords": {
                    "type": "boolean",
                    "description": "Whether to remove stopwords",
                    "default": True,
                },
                "add_context_markers": {
                    "type": "boolean",
                    "description": "Whether to add context markers",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "optimized_query": {"type": "string"},  # Optimized query
                    "optimization_steps": {"type": "array"},  # Steps applied
                    "processing_time": {"type": "number"},
                },
            },
        )

    def expand_query(
        self,
        query: str,
        domain: Optional[str] = None,
        strategies: Optional[List[str]] = None,
        max_terms: int = None,
    ) -> Dict[str, Any]:
        """
        Expand a query with related terms to improve retrieval.

        Args:
            query: Original query to expand
            domain: Domain context for query expansion
            strategies: Expansion strategies to use
            max_terms: Maximum number of terms to add in expansion

        Returns:
            Dictionary with expanded query
        """
        start_time = time.time()

        try:
            if not query:
                return {
                    "error": "Query cannot be empty",
                    "processing_time": time.time() - start_time,
                }

            # Set default values
            max_terms = max_terms or self.max_expansion_terms
            strategies = strategies or self.expansion_strategies

            # Initialize result
            expansion_terms = []
            strategies_applied = []
            original_query = query.strip()

            # Apply each expansion strategy
            if "synonyms" in strategies:
                synonym_result = self.add_synonym_expansion(
                    query=original_query,
                    max_synonyms_per_term=min(
                        2, max_terms
                    ),  # Limit synonyms based on max_terms
                )

                if not synonym_result.get("error"):
                    # Extract only the new terms added, not the full expanded query
                    new_synonyms = []
                    for term, synonyms in synonym_result.get(
                        "synonym_mappings", {}
                    ).items():
                        new_synonyms.extend(synonyms)

                    # Add unique terms only
                    for term in new_synonyms:
                        if (
                            term not in expansion_terms
                            and term.lower() not in original_query.lower()
                        ):
                            expansion_terms.append(term)
                            if len(expansion_terms) >= max_terms:
                                break

                    strategies_applied.append("synonyms")

            if "domain_terms" in strategies and domain:
                domain_result = self.enhance_domain_specificity(
                    query=original_query,
                    domain=domain,
                    max_terms=min(
                        max_terms - len(expansion_terms), 3
                    ),  # Adjust based on remaining terms
                )

                if not domain_result.get("error"):
                    # Add unique domain terms
                    for term in domain_result.get("added_terms", []):
                        if (
                            term not in expansion_terms
                            and term.lower() not in original_query.lower()
                        ):
                            expansion_terms.append(term)
                            if len(expansion_terms) >= max_terms:
                                break

                    strategies_applied.append("domain_terms")

            if "query_decomposition" in strategies:
                decompose_result = self.decompose_query(
                    query=original_query,
                    max_sub_queries=2,  # Just get a couple of sub-queries
                )

                if not decompose_result.get("error"):
                    # Extract key terms from sub-queries
                    for sub_query in decompose_result.get("sub_queries", []):
                        # Extract noun phrases or key terms from sub-query
                        key_terms = self._extract_key_terms(sub_query)
                        for term in key_terms:
                            if (
                                term not in expansion_terms
                                and term.lower() not in original_query.lower()
                            ):
                                expansion_terms.append(term)
                                if len(expansion_terms) >= max_terms:
                                    break

                    strategies_applied.append("query_decomposition")

            # Limit to max_terms
            expansion_terms = expansion_terms[:max_terms]

            # Construct expanded query
            if expansion_terms:
                expanded_query = f"{original_query} {' '.join(expansion_terms)}"
            else:
                expanded_query = original_query

            return {
                "expanded_query": expanded_query,
                "expansion_terms": expansion_terms,
                "strategies_applied": strategies_applied,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error expanding query: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def decompose_query(self, query: str, max_sub_queries: int = 3) -> Dict[str, Any]:
        """
        Break down a complex query into simpler sub-queries.

        Args:
            query: Complex query to decompose
            max_sub_queries: Maximum number of sub-queries to generate

        Returns:
            Dictionary with sub-queries
        """
        start_time = time.time()

        try:
            if not query:
                return {
                    "error": "Query cannot be empty",
                    "processing_time": time.time() - start_time,
                }

            # Simple rule-based decomposition
            # 1. Split on conjunctions and question markers
            split_patterns = [
                r" and ",
                r" or ",
                r" but ",
                r" as well as ",
                r"\?",
                r";",
                r",",
            ]

            parts = [query]
            for pattern in split_patterns:
                new_parts = []
                for part in parts:
                    split_result = re.split(pattern, part, flags=re.IGNORECASE)
                    new_parts.extend([p.strip() for p in split_result if p.strip()])
                parts = new_parts
                if (
                    len(parts) >= max_sub_queries * 2
                ):  # Get more than needed for filtering
                    break

            # 2. Filter out very short parts and duplicates
            filtered_parts = []
            seen = set()
            for part in parts:
                if len(part.split()) >= 3 and part.lower() not in seen:
                    filtered_parts.append(part)
                    seen.add(part.lower())

            # 3. Identify the main focus (usually the longest or first part)
            main_focus = query
            if filtered_parts:
                # Use the longest part as the main focus
                main_focus = max(filtered_parts, key=len)

            # 4. Limit to max_sub_queries
            sub_queries = filtered_parts[:max_sub_queries]

            # 5. If we couldn't decompose effectively, create alternative phrasings
            if len(sub_queries) <= 1:
                # Try to generate alternative phrasings
                if query.lower().startswith("how"):
                    sub_queries.append(f"What is the process for {query[3:].strip()}")
                elif query.lower().startswith("what"):
                    sub_queries.append(f"Explain {query[4:].strip()}")
                elif query.lower().startswith("why"):
                    sub_queries.append(f"Reasons for {query[3:].strip()}")

                # Add a more specific version if possible
                words = query.split()
                if len(words) > 5:
                    sub_queries.append(" ".join(words[: len(words) // 2]))
                    sub_queries.append(" ".join(words[len(words) // 2 :]))

            # Remove duplicates again and limit
            final_sub_queries = []
            seen = set()
            for q in sub_queries:
                if q.lower() not in seen and q.strip():
                    final_sub_queries.append(q)
                    seen.add(q.lower())
                    if len(final_sub_queries) >= max_sub_queries:
                        break

            return {
                "sub_queries": final_sub_queries,
                "main_focus": main_focus,
                "original_query": query,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error decomposing query: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def enhance_domain_specificity(
        self, query: str, domain: str, max_terms: int = 3
    ) -> Dict[str, Any]:
        """
        Enhance a query with domain-specific terminology.

        Args:
            query: Original query to enhance
            domain: Domain for terminology enhancement
            max_terms: Maximum number of domain terms to add

        Returns:
            Dictionary with domain-enhanced query
        """
        start_time = time.time()

        try:
            if not query:
                return {
                    "error": "Query cannot be empty",
                    "processing_time": time.time() - start_time,
                }

            if not domain:
                return {
                    "error": "Domain must be specified",
                    "processing_time": time.time() - start_time,
                }

            # Get domain-specific terms
            domain_terms = self.domain_specific_terms.get(domain.lower(), [])
            if not domain_terms:
                return {
                    "error": f"No terms available for domain: {domain}",
                    "processing_time": time.time() - start_time,
                }

            # Find terms that are not already in the query
            query_lower = query.lower()
            relevant_terms = []

            for term in domain_terms:
                if term.lower() not in query_lower:
                    # Calculate relevance based on query content
                    relevance = self._calculate_term_relevance(term, query)
                    if relevance >= self.min_term_relevance:
                        relevant_terms.append((term, relevance))

            # Sort by relevance and take top terms
            relevant_terms.sort(key=lambda x: x[1], reverse=True)
            selected_terms = [term for term, _ in relevant_terms[:max_terms]]

            # Construct enhanced query
            if selected_terms:
                enhanced_query = f"{query} {' '.join(selected_terms)}"
            else:
                enhanced_query = query

            return {
                "enhanced_query": enhanced_query,
                "added_terms": selected_terms,
                "domain": domain,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error enhancing domain specificity: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def add_synonym_expansion(
        self, query: str, max_synonyms_per_term: int = 2
    ) -> Dict[str, Any]:
        """
        Expand a query with synonyms of key terms.

        Args:
            query: Original query to expand with synonyms
            max_synonyms_per_term: Maximum number of synonyms to add per term

        Returns:
            Dictionary with synonym-expanded query
        """
        start_time = time.time()

        try:
            if not query:
                return {
                    "error": "Query cannot be empty",
                    "processing_time": time.time() - start_time,
                }

            # Extract key terms from the query
            key_terms = self._extract_key_terms(query)

            # Find synonyms for key terms
            synonym_mappings = {}
            query_words = set(query.lower().split())

            for term in key_terms:
                term_lower = term.lower()
                # Check if we have synonyms for this term
                for base_term, synonyms in self.synonym_mappings.items():
                    if (
                        base_term.lower() == term_lower
                        or base_term.lower() in term_lower.split()
                    ):
                        # Filter out synonyms already in the query
                        filtered_synonyms = [
                            s for s in synonyms if s.lower() not in query_words
                        ]
                        if filtered_synonyms:
                            # Limit to max_synonyms_per_term
                            synonym_mappings[term] = filtered_synonyms[
                                :max_synonyms_per_term
                            ]
                            break

            # Construct expanded query
            expanded_parts = [query]
            for term, synonyms in synonym_mappings.items():
                expanded_parts.extend(synonyms)

            expanded_query = " ".join(expanded_parts)

            return {
                "expanded_query": expanded_query,
                "synonym_mappings": synonym_mappings,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error adding synonym expansion: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def optimize_for_vector_search(
        self,
        query: str,
        remove_stopwords: bool = True,
        add_context_markers: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize a query for vector-based semantic search.

        Args:
            query: Original query to optimize
            remove_stopwords: Whether to remove stopwords
            add_context_markers: Whether to add context markers

        Returns:
            Dictionary with optimized query
        """
        start_time = time.time()

        try:
            if not query:
                return {
                    "error": "Query cannot be empty",
                    "processing_time": time.time() - start_time,
                }

            optimization_steps = []
            optimized_query = query

            # 1. Remove stopwords if requested
            if remove_stopwords:
                stopwords = [
                    "a",
                    "an",
                    "the",
                    "and",
                    "or",
                    "but",
                    "if",
                    "then",
                    "else",
                    "when",
                    "at",
                    "from",
                    "by",
                    "for",
                    "with",
                    "about",
                    "against",
                    "between",
                    "into",
                    "through",
                    "during",
                    "before",
                    "after",
                    "above",
                    "below",
                    "to",
                    "of",
                    "in",
                    "on",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "being",
                ]

                words = optimized_query.split()
                filtered_words = [
                    word for word in words if word.lower() not in stopwords
                ]

                if filtered_words:  # Ensure we don't end up with an empty query
                    optimized_query = " ".join(filtered_words)
                    optimization_steps.append("removed_stopwords")

            # 2. Add context markers if requested
            if add_context_markers:
                # Add markers to help with semantic understanding
                optimized_query = f"query: {optimized_query}"
                optimization_steps.append("added_context_markers")

            # 3. Ensure query is well-formed
            # Remove excessive whitespace
            optimized_query = re.sub(r"\s+", " ", optimized_query).strip()
            optimization_steps.append("normalized_whitespace")

            # 4. Capitalize first letter for better semantic parsing
            if optimized_query:
                optimized_query = optimized_query[0].upper() + optimized_query[1:]
                optimization_steps.append("capitalized_first_letter")

            return {
                "optimized_query": optimized_query,
                "original_query": query,
                "optimization_steps": optimization_steps,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error optimizing for vector search: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text using simple heuristics."""
        # This is a simplified implementation
        # In a production system, you might use NLP libraries like spaCy

        # Remove punctuation and convert to lowercase
        text = re.sub(r"[^\w\s]", " ", text.lower())

        # Split into words
        words = text.split()

        # Remove common stopwords
        stopwords = [
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "at",
            "from",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "of",
            "in",
            "on",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        ]

        filtered_words = [
            word for word in words if word not in stopwords and len(word) > 2
        ]

        # Find potential multi-word terms (simple approach)
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in stopwords or words[i + 1] not in stopwords:
                bigram = f"{words[i]} {words[i + 1]}"
                if len(bigram) > 5:  # Avoid very short bigrams
                    bigrams.append(bigram)

        # Combine single words and bigrams, prioritizing longer terms
        key_terms = bigrams + filtered_words

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def _calculate_term_relevance(self, term: str, query: str) -> float:
        """Calculate relevance of a term to a query using simple heuristics."""
        # This is a simplified implementation
        # In a production system, you might use more sophisticated methods

        query_words = set(re.sub(r"[^\w\s]", " ", query.lower()).split())
        term_words = set(re.sub(r"[^\w\s]", " ", term.lower()).split())

        # Check for direct word overlap
        overlap = query_words.intersection(term_words)
        if overlap:
            return 0.8  # High relevance if words overlap

        # Check for substring matches
        for qw in query_words:
            for tw in term_words:
                if len(qw) > 3 and len(tw) > 3:
                    if qw in tw or tw in qw:
                        return 0.6  # Medium relevance for substring matches

        # Default relevance based on term length (prefer shorter, more focused terms)
        return 0.5 if len(term) < 15 else 0.4
