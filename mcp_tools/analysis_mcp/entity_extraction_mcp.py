"""
Entity Extraction MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
entity extraction capabilities for financial text, identifying companies,
mapping them to stock tickers, and detecting relationships between entities.
"""

import os
import json
import re
import time
from typing import Dict, List, Any, Optional
import difflib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For entity recognition
try:
    import spacy

    HAVE_SPACY = True
except ImportError:
    HAVE_SPACY = False

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-entity-extraction",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class EntityExtractionMCP(BaseMCPServer):
    """
    MCP server for extracting financial entities from text.

    This tool identifies companies, people, and financial entities in text
    and maps them to stock tickers and other structured data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Entity Extraction MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - spacy_model: SpaCy model to use (default: en_core_web_sm)
                - ticker_mapping_path: Path to ticker to company name mapping
                - custom_entities_path: Path to custom financial entity list
                - cache_dir: Directory for caching models
        """
        super().__init__(name="entity_extraction_mcp", config=config)

        # Set default configurations
        self.spacy_model_name = self.config.get("spacy_model", "en_core_web_sm")
        self.ticker_mapping_path = self.config.get(
            "ticker_mapping_path",
            os.path.join(os.path.dirname(__file__), "data/ticker_to_name.json"),
        )
        self.custom_entities_path = self.config.get(
            "custom_entities_path",
            os.path.join(os.path.dirname(__file__), "data/financial_entities.json"),
        )
        self.cache_dir = self.config.get("cache_dir", "./model_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load models and data
        self._load_spacy_model()
        self._load_ticker_mapping()
        self._load_custom_entities()

        # Register tools
        self._register_tools()

    def _load_spacy_model(self):
        """Load and configure SpaCy model."""
        if not HAVE_SPACY:
            self.logger.warning(
                "SpaCy not available. Entity extraction will be limited."
            )
            self.nlp = None
            return

        try:
            # Load SpaCy model
            self.logger.info(f"Loading SpaCy model: {self.spacy_model_name}")
            try:
                self.nlp = spacy.load(self.spacy_model_name)
                self.logger.info(
                    f"Successfully loaded SpaCy model: {self.spacy_model_name}"
                )
            except OSError:
                self.logger.warning(
                    f"SpaCy model {self.spacy_model_name} not found, downloading..."
                )
                spacy.cli.download(self.spacy_model_name)
                self.nlp = spacy.load(self.spacy_model_name)
                self.logger.info(
                    f"Successfully downloaded and loaded SpaCy model: {self.spacy_model_name}"
                )

            # Add pipeline components for financial entity recognition
            # Note: In a real implementation, we might add a custom component
            # for financial entity recognition

        except Exception as e:
            self.logger.error(f"Error loading SpaCy model: {e}")
            self.nlp = None

    def _load_ticker_mapping(self):
        """Load mapping between ticker symbols and company names."""
        self.ticker_to_name = {}
        self.name_to_ticker = {}

        # Check if ticker mapping file exists, otherwise use default mappings
        if os.path.exists(self.ticker_mapping_path):
            try:
                with open(self.ticker_mapping_path, "r") as f:
                    self.ticker_to_name = json.load(f)

                # Create reverse mapping
                for ticker, company_info in self.ticker_to_name.items():
                    if isinstance(company_info, dict) and "name" in company_info:
                        company_name = company_info["name"].lower()
                        self.name_to_ticker[company_name] = ticker
                    else:
                        self.name_to_ticker[company_info.lower()] = ticker

                self.logger.info(f"Loaded {len(self.ticker_to_name)} ticker mappings")

            except Exception as e:
                self.logger.error(f"Error loading ticker mapping: {e}")
                self._create_default_ticker_mapping()
        else:
            self.logger.warning(
                f"Ticker mapping file not found: {self.ticker_mapping_path}"
            )
            self._create_default_ticker_mapping()

    def _create_default_ticker_mapping(self):
        """Create a default ticker mapping with major companies."""
        self.ticker_to_name = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
            "MSFT": {"name": "Microsoft Corporation", "sector": "Technology"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
            "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
            "META": {"name": "Meta Platforms Inc.", "sector": "Technology"},
            "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Cyclical"},
            "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology"},
            "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
            "V": {"name": "Visa Inc.", "sector": "Financial Services"},
            "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive"},
        }

        # Create reverse mapping
        self.name_to_ticker = {
            info["name"].lower(): ticker for ticker, info in self.ticker_to_name.items()
        }

        # Add common name variations
        self.name_to_ticker.update(
            {
                "apple": "AAPL",
                "microsoft": "MSFT",
                "google": "GOOGL",
                "alphabet": "GOOGL",
                "amazon": "AMZN",
                "meta": "META",
                "facebook": "META",
                "tesla": "TSLA",
                "nvidia": "NVDA",
                "jpmorgan": "JPM",
                "jpmorgan chase": "JPM",
                "visa": "V",
                "walmart": "WMT",
            }
        )

        self.logger.info("Created default ticker mapping with 10 major companies")

    def _load_custom_entities(self):
        """Load custom financial entity lists."""
        self.financial_entities = {
            "financial_terms": [],
            "regulatory_bodies": [],
            "economic_indicators": [],
            "financial_events": [],
        }

        if os.path.exists(self.custom_entities_path):
            try:
                with open(self.custom_entities_path, "r") as f:
                    self.financial_entities = json.load(f)
                self.logger.info(
                    f"Loaded custom financial entities from {self.custom_entities_path}"
                )
            except Exception as e:
                self.logger.error(f"Error loading custom entities: {e}")
                self._create_default_financial_entities()
        else:
            self.logger.warning(
                f"Custom entities file not found: {self.custom_entities_path}"
            )
            self._create_default_financial_entities()

    def _create_default_financial_entities(self):
        """Create default financial entity lists."""
        self.financial_entities = {
            "financial_terms": [
                "earnings",
                "revenue",
                "profit",
                "loss",
                "dividend",
                "merger",
                "acquisition",
                "IPO",
                "quarterly results",
                "fiscal year",
            ],
            "regulatory_bodies": [
                "SEC",
                "Federal Reserve",
                "Fed",
                "FDIC",
                "OCC",
                "CFTC",
                "FINRA",
            ],
            "economic_indicators": [
                "GDP",
                "inflation",
                "unemployment",
                "interest rate",
                "CPI",
                "consumer confidence",
                "PMI",
                "retail sales",
                "housing starts",
            ],
            "financial_events": [
                "earnings call",
                "shareholder meeting",
                "stock split",
                "dividend announcement",
                "buyback",
                "guidance update",
            ],
        }

        self.logger.info("Created default financial entity lists")

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "extract_entities",
            self.extract_entities,
            "Extract entities (companies, people, locations, etc.) from text",
            {
                "text": {
                    "type": "string",
                    "description": "Text to extract entities from",
                },
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of entities to extract (default: all)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "entities": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "map_to_tickers",
            self.map_to_tickers,
            "Map company names to ticker symbols",
            {
                "company_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of company names to map to tickers",
                },
                "fuzzy_match": {
                    "type": "boolean",
                    "description": "Whether to use fuzzy matching for company names",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "mappings": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "extract_from_news",
            self.extract_from_news,
            "Extract entities from a news article with title and content",
            {
                "title": {"type": "string", "description": "Title of the news article"},
                "content": {
                    "type": "string",
                    "description": "Content of the news article",
                },
            },
            {
                "type": "object",
                "properties": {
                    "companies": {"type": "array"},
                    "tickers": {"type": "array"},
                    "people": {"type": "array"},
                    "financial_terms": {"type": "array"},
                    "regulatory_bodies": {"type": "array"},
                    "economic_indicators": {"type": "array"},
                    "locations": {"type": "array"},
                    "main_entities": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "detect_relationships",
            self.detect_relationships,
            "Detect relationships between entities in text",
            {
                "text": {
                    "type": "string",
                    "description": "Text to analyze for entity relationships",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entities to focus on (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "relationships": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

    def extract_entities(
        self, text: str, entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text.

        Args:
            text: Text to analyze
            entity_types: Optional list of entity types to extract
                        (e.g., "ORG", "PERSON", "GPE")

        Returns:
            Dictionary of extracted entities grouped by type
        """
        start_time = time.time()

        if not self.nlp:
            return {
                "entities": {},
                "error": "SpaCy model not loaded",
                "processing_time": 0.0,
            }

        try:
            # Process the text with SpaCy
            doc = self.nlp(text)

            # Filter by entity types if specified
            if entity_types:
                entity_types = [t.upper() for t in entity_types]
                entities = {
                    entity_type: [
                        {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
                        for ent in doc.ents
                        if ent.label_ == entity_type
                    ]
                    for entity_type in entity_types
                }
            else:
                # Group by entity type
                entities = {}
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []

                    entities[ent.label_].append(
                        {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
                    )

            # Add companies from our ticker mapping that appear in the text
            companies = self._extract_companies_from_text(text)
            if companies:
                entities["COMPANY"] = entities.get("COMPANY", []) + companies

                # Remove duplicates
                if "COMPANY" in entities:
                    unique_companies = {}
                    for company in entities["COMPANY"]:
                        unique_companies[company["text"].lower()] = company
                    entities["COMPANY"] = list(unique_companies.values())

            # Add financial terms
            financial_terms = self._extract_financial_terms(text)
            if financial_terms:
                entities["FINANCIAL_TERM"] = financial_terms

            processing_time = time.time() - start_time

            return {"entities": entities, "processing_time": processing_time}

        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return {
                "entities": {},
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _extract_companies_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract company names from text using the ticker mapping."""
        companies = []
        text_lower = text.lower()

        # Look for exact matches of company names
        for company_name, ticker in self.name_to_ticker.items():
            # Find all occurrences of the company name
            company_name_lower = company_name.lower()

            # Use regex to find word boundary matches
            pattern = r"\b" + re.escape(company_name_lower) + r"\b"
            for match in re.finditer(pattern, text_lower):
                start = match.start()
                end = match.end()

                companies.append(
                    {
                        "text": text[start:end],  # Use original case from text
                        "start": start,
                        "end": end,
                        "ticker": ticker,
                    }
                )

        return companies

    def _extract_financial_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial terms from text."""
        terms = []
        text_lower = text.lower()

        # Look for financial terms
        for term_type, term_list in self.financial_entities.items():
            for term in term_list:
                term_lower = term.lower()

                # Use regex to find word boundary matches
                pattern = r"\b" + re.escape(term_lower) + r"\b"
                for match in re.finditer(pattern, text_lower):
                    start = match.start()
                    end = match.end()

                    terms.append(
                        {
                            "text": text[start:end],  # Use original case from text
                            "start": start,
                            "end": end,
                            "term_type": term_type,
                        }
                    )

        return terms

    def map_to_tickers(
        self, company_names: List[str], fuzzy_match: bool = False
    ) -> Dict[str, Any]:
        """
        Map company names to ticker symbols.

        Args:
            company_names: List of company names to map
            fuzzy_match: Whether to use fuzzy matching

        Returns:
            Mapping of company names to tickers
        """
        start_time = time.time()

        mappings = {}

        for company in company_names:
            company_lower = company.lower()

            # Try exact match first
            if company_lower in self.name_to_ticker:
                ticker = self.name_to_ticker[company_lower]
                company_info = self.ticker_to_name.get(ticker, {})
                if isinstance(company_info, str):
                    company_info = {"name": company_info}

                mappings[company] = {
                    "ticker": ticker,
                    "match_type": "exact",
                    "confidence": 1.0,
                    "official_name": company_info.get("name", ""),
                    "sector": company_info.get("sector", ""),
                }

            # Try fuzzy match if enabled
            elif fuzzy_match:
                best_match = None
                best_ratio = 0.0

                for name in self.name_to_ticker.keys():
                    ratio = difflib.SequenceMatcher(None, company_lower, name).ratio()
                    if ratio > 0.8 and ratio > best_ratio:  # 80% similarity threshold
                        best_match = name
                        best_ratio = ratio

                if best_match:
                    ticker = self.name_to_ticker[best_match]
                    company_info = self.ticker_to_name.get(ticker, {})
                    if isinstance(company_info, str):
                        company_info = {"name": company_info}

                    mappings[company] = {
                        "ticker": ticker,
                        "match_type": "fuzzy",
                        "confidence": best_ratio,
                        "official_name": company_info.get("name", ""),
                        "sector": company_info.get("sector", ""),
                    }
                else:
                    mappings[company] = {
                        "ticker": None,
                        "match_type": "no_match",
                        "confidence": 0.0,
                    }
            else:
                mappings[company] = {
                    "ticker": None,
                    "match_type": "no_match",
                    "confidence": 0.0,
                }

        return {"mappings": mappings, "processing_time": time.time() - start_time}

    def extract_from_news(self, title: str, content: str) -> Dict[str, Any]:
        """
        Extract entities from a news article.

        Args:
            title: Article title
            content: Article content

        Returns:
            Extracted entities categorized
        """
        start_time = time.time()

        # Combine title and content for analysis, but weighted differently
        # Title entities are likely more important
        full_text = f"{title}\n\n{content}"

        # Extract all entities
        result = self.extract_entities(full_text)
        entities = result.get("entities", {})

        # Organize results by category
        companies = []
        tickers = set()
        people = []
        locations = []
        financial_terms = []
        regulatory_bodies = []
        economic_indicators = []

        # Process companies (ORG or COMPANY)
        for entity_type in ["ORG", "COMPANY"]:
            if entity_type in entities:
                for entity in entities[entity_type]:
                    company_name = entity["text"]

                    # Try to map to ticker
                    mapping_result = self.map_to_tickers(
                        [company_name], fuzzy_match=True
                    )
                    mapping = mapping_result.get("mappings", {}).get(company_name, {})

                    if mapping.get("ticker"):
                        # Add ticker information
                        company = {
                            "name": company_name,
                            "ticker": mapping.get("ticker"),
                            "official_name": mapping.get("official_name", company_name),
                            "sector": mapping.get("sector", ""),
                            "confidence": mapping.get("confidence", 0.0),
                        }
                        companies.append(company)
                        tickers.add(mapping.get("ticker"))

        # Process people (PERSON)
        if "PERSON" in entities:
            people = [{"name": entity["text"]} for entity in entities["PERSON"]]

        # Process locations (GPE, LOC)
        for entity_type in ["GPE", "LOC"]:
            if entity_type in entities:
                locations.extend(
                    [
                        {"name": entity["text"], "type": entity_type}
                        for entity in entities[entity_type]
                    ]
                )

        # Process financial terms
        if "FINANCIAL_TERM" in entities:
            for entity in entities["FINANCIAL_TERM"]:
                term_type = entity.get("term_type", "")
                term = {"term": entity["text"], "type": term_type}

                financial_terms.append(term)

                # Categorize by term type
                if term_type == "regulatory_bodies":
                    regulatory_bodies.append(term)
                elif term_type == "economic_indicators":
                    economic_indicators.append(term)

        # Determine main entities (those in title)
        title_result = self.extract_entities(title)
        title_entities = title_result.get("entities", {})

        main_entities = []
        for entity_list in title_entities.values():
            for entity in entity_list:
                main_entities.append(entity["text"])

        return {
            "companies": companies,
            "tickers": list(tickers),
            "people": people,
            "locations": locations,
            "financial_terms": financial_terms,
            "regulatory_bodies": regulatory_bodies,
            "economic_indicators": economic_indicators,
            "main_entities": main_entities,
            "processing_time": time.time() - start_time,
        }

    def detect_relationships(
        self, text: str, entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect relationships between entities in text.

        Args:
            text: Text to analyze
            entities: Optional list of entities to focus on

        Returns:
            List of detected relationships between entities
        """
        start_time = time.time()

        if not self.nlp:
            return {
                "relationships": [],
                "error": "SpaCy model not loaded",
                "processing_time": 0.0,
            }

        try:
            # Process the text with SpaCy
            doc = self.nlp(text)

            # First, identify all entities in the text
            entity_mentions = {}

            # Track entities from SpaCy's NER
            for ent in doc.ents:
                entity_text = ent.text
                if entities and entity_text not in entities:
                    continue

                if entity_text not in entity_mentions:
                    entity_mentions[entity_text] = {"type": ent.label_, "mentions": []}

                entity_mentions[entity_text]["mentions"].append(
                    {
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "sentence_id": next(
                            (
                                i
                                for i, sent in enumerate(doc.sents)
                                if sent.start_char <= ent.start_char
                                and sent.end_char >= ent.end_char
                            ),
                            0,
                        ),
                    }
                )

            # Also track company names from our ticker mappings
            for company_name in self.name_to_ticker.keys():
                if entities and company_name not in entities:
                    continue

                company_name_lower = company_name.lower()
                text_lower = text.lower()

                pattern = r"\b" + re.escape(company_name_lower) + r"\b"
                for match in re.finditer(pattern, text_lower):
                    start = match.start()
                    end = match.end()

                    if company_name not in entity_mentions:
                        entity_mentions[company_name] = {
                            "type": "COMPANY",
                            "mentions": [],
                        }

                    # Find which sentence this belongs to
                    sentence_id = next(
                        (
                            i
                            for i, sent in enumerate(doc.sents)
                            if sent.start_char <= start and sent.end_char >= end
                        ),
                        0,
                    )

                    entity_mentions[company_name]["mentions"].append(
                        {"start": start, "end": end, "sentence_id": sentence_id}
                    )

            # Find relationships between entities
            relationships = []

            # Entities in the same sentence might be related
            for entity1, info1 in entity_mentions.items():
                for mention1 in info1["mentions"]:
                    sentence_id = mention1["sentence_id"]

                    # Find other entities in the same sentence
                    for entity2, info2 in entity_mentions.items():
                        if entity1 == entity2:
                            continue

                        for mention2 in info2["mentions"]:
                            if mention2["sentence_id"] == sentence_id:
                                # Found a potential relationship
                                # Extract the sentence for context
                                for sent_id, sent in enumerate(doc.sents):
                                    if sent_id == sentence_id:
                                        relationships.append(
                                            {
                                                "entity1": entity1,
                                                "entity1_type": info1["type"],
                                                "entity2": entity2,
                                                "entity2_type": info2["type"],
                                                "sentence": sent.text,
                                                "confidence": 0.7,  # Placeholder confidence
                                            }
                                        )
                                        break

            # Remove duplicates (same entity pair in different orders)
            unique_relationships = {}
            for rel in relationships:
                # Create a unique key for each entity pair
                entities_key = tuple(sorted([rel["entity1"], rel["entity2"]]))
                if entities_key not in unique_relationships:
                    unique_relationships[entities_key] = rel

            return {
                "relationships": list(unique_relationships.values()),
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error detecting relationships: {e}")
            return {
                "relationships": [],
                "error": str(e),
                "processing_time": time.time() - start_time,
            }


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {"spacy_model": "en_core_web_sm"}

    # Create and start the server
    server = EntityExtractionMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("EntityExtractionMCP server started")

    # Example usage
    result = server.extract_from_news(
        "Apple Reports Record Earnings, Partners with Microsoft on AI Initiative",
        "Apple Inc. announced today that it has exceeded analyst expectations with a record quarterly revenue of $90.1 billion. CEO Tim Cook highlighted strong iPhone sales and growing services revenue. The company also revealed a new partnership with Microsoft Corporation on artificial intelligence initiatives, which will be led by their CTO in Cupertino, California.",
    )
    print(f"Entity extraction result: {json.dumps(result, indent=2)}")
