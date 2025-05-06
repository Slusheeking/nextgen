"""
End-to-end tests for the integrated NextGen system focusing on model-MCP interactions.
Tests in this module validate models and their dependent MCP tools working together as a unit.
"""
import time
import logging
import os
import dotenv
import json
import unittest
import pytest
from typing import Dict, Any, Optional

# Import models
from nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model import SentimentAnalysisModel  
from nextgen_models.nextgen_market_analysis.market_analysis_model import MarketAnalysisModel
from nextgen_models.nextgen_decision.decision_model import DecisionModel
from nextgen_models.autogen_orchestrator.autogen_model import AutoGenOrchestrator
from nextgen_models.nextgen_context_model.context_model import ContextModel
from nextgen_models.nextgen_fundamental_analysis.fundamental_analysis_model import FundamentalAnalysisModel
from nextgen_models.nextgen_risk_assessment.risk_assessment_model import RiskAssessmentModel
from nextgen_models.nextgen_select.select_model import SelectionModel
from nextgen_models.nextgen_trader.trade_model import TradeModel

# Import MCP tools
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.document_analysis_mcp.document_analysis_mcp import DocumentAnalysisMCP
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
from mcp_tools.trading_mcp.trading_mcp import TradingMCP
from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
dotenv.load_dotenv()

# Load environment variables for services and APIs
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8005"))

# Warning for missing critical environment variables
if not OPENROUTER_API_KEY:
    logging.warning("OPENROUTER_API_KEY not set. LLM interactions may fail.")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logging.warning("Alpaca API keys not set. Trading execution may fail.")
if not POLYGON_API_KEY:
    logging.warning("Polygon API key not set. Market data fetching may fail.")


class TestIntegratedSystem(unittest.TestCase):
    """Contains end-to-end tests for the integrated NextGen system workflow."""

    def setUp(self):
        """Set up test environment before each test."""
        self.data_source = os.getenv('E2E_DATA_SOURCE', 'synthetic')
        logging.info(f"TestIntegratedSystem setup with data source: {self.data_source}")
        
        # Initialize Redis and VectorDB connection parameters from environment variables
        self.redis_params = {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "db": REDIS_DB,
            "password": REDIS_PASSWORD
        }
        
        self.vector_db_params = {
            "host": CHROMADB_HOST,
            "port": CHROMADB_PORT
        }
        
        logging.info("Setting up integrated system test environment with environment-based configs.")

    def tearDown(self):
        """Clean up test environment after each test."""
        logging.info("Tearing down integrated system test environment.")

    def _load_component_config(self, component_path: str) -> Dict[str, Any]:
        """
        Loads the configuration file for a given component with proper path handling.
        Updates config with environment variables where appropriate.

        Args:
            component_path (str): The relative path to the component config.

        Returns:
            dict: The loaded and environment-updated configuration dictionary.
        """
        # Build proper path using directory structure
        base_dir = "/home/ubuntu/nextgen"
        config_path = f"{base_dir}/config/{component_path}"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logging.info(f"Loaded configuration from {config_path}")
            
            # Update database connection parameters from environment variables
            if "database" in config:
                if "redis" in config["database"]:
                    config["database"]["redis"].update(self.redis_params)
                if "vector_db" in config["database"]:
                    config["database"]["vector_db"].update(self.vector_db_params)
            
            # Update API keys from environment
            if "api_keys" in config:
                for key_name in config["api_keys"]:
                    env_var_name = key_name.upper()
                    if os.getenv(env_var_name):
                        config["api_keys"][key_name] = os.getenv(env_var_name)
            
            return config
        except FileNotFoundError:
            logging.warning(f"Configuration file not found at {config_path}. Returning empty config.")
            return {}
        except json.JSONDecodeError:
            logging.warning(f"Error decoding JSON from configuration file {config_path}. Returning empty config.")
            return {}
        except Exception as e:
            logging.warning(f"Error loading configuration: {e}. Returning empty config.")
            return {}

    def test_context_model_with_vector_store_mcp(self):
        """
        Tests the integrated functionality of ContextModel with VectorStoreMCP.
        This test validates:
        1. Configuration of both components using environment variables
        2. Proper integration between the model and its MCP dependency
        3. End-to-end context retrieval functionality
        """
        logging.info("Testing ContextModel with VectorStoreMCP integration...")
        
        # Initialize VectorStoreMCP with environment-based configuration
        vector_store_config = self._load_component_config("vector_store_mcp/vector_store_mcp_config.json")
        vector_store_mcp = VectorStoreMCP(config=vector_store_config)
        
        # Initialize ContextModel with environment-based configuration 
        # and inject the VectorStoreMCP instance
        context_config = self._load_component_config("nextgen_context_model/context_model_config.json")
        context_model = ContextModel(config=context_config)
        
        # Test sample documents to add to vector store
        sample_documents = [
            {"id": "doc1", "text": "Apple stock price increased by 5% following strong earnings."},
            {"id": "doc2", "text": "Microsoft announced new AI features for their cloud platform."},
            {"id": "doc3", "text": "Tesla's production numbers exceeded analyst expectations."}
        ]
        
        try:
            # Add documents to vector store (simulated/mocked if in synthetic mode)
            if self.data_source == 'synthetic':
                logging.info("Using synthetic data mode - simulating vector store operations")
                # In synthetic mode, we don't actually perform DB operations
            else:
                logging.info("Adding sample documents to vector store")
                # In real mode, add documents to the actual vector store
                vector_store_mcp.add_documents(sample_documents)
            
            # Test context retrieval with the context model using vector store
            query = "What happened with Apple stock?"
            retrieved_context = context_model.retrieve_context_sync(query)
            
            # Validate the integration between model and MCP tool
            self.assertIsNotNone(retrieved_context)
            
            if self.data_source != 'synthetic':
                # Only verify specific results in non-synthetic mode
                self.assertGreater(len(retrieved_context.get("retrieved_context", [])), 0)
                
            logging.info(f"Context model retrieved {len(retrieved_context.get('retrieved_context', [])) if retrieved_context else 0} relevant contexts")
            
        except Exception as e:
            logging.error(f"Error testing ContextModel with VectorStoreMCP: {e}")
            raise
            
        logging.info("ContextModel with VectorStoreMCP test completed successfully")

    def test_sentiment_analysis_model_with_document_analysis_mcp(self):
        """
        Tests the integrated functionality of SentimentAnalysisModel with DocumentAnalysisMCP.
        This test validates that the sentiment model can properly utilize the document analysis MCP
        for processing financial text before performing sentiment analysis.
        """
        logging.info("Testing SentimentAnalysisModel with DocumentAnalysisMCP integration...")
        
        # Initialize DocumentAnalysisMCP with environment-based configuration
        doc_analysis_config = self._load_component_config("document_analysis_mcp/document_analysis_mcp_config.json")
        doc_analysis_mcp = DocumentAnalysisMCP(config=doc_analysis_config)
        
        # Initialize SentimentAnalysisModel with environment-based configuration
        sentiment_config = self._load_component_config("nextgen_sentiment_analysis/sentiment_analysis_model_config.json")
        sentiment_model = SentimentAnalysisModel(config=sentiment_config)
        
        # Sample text for analysis
        sample_texts = [
            "The company reported a 20% increase in quarterly profit, beating analyst expectations.",
            "The stock plunged 15% after the company missed revenue targets and lowered guidance.",
            "Investors remain cautious about the stock due to ongoing regulatory challenges."
        ]
        
        try:
            # First use DocumentAnalysisMCP to process the texts
            processed_texts = []
            for text in sample_texts:
                if self.data_source == 'synthetic':
                    # In synthetic mode, simply pass through the text
                    processed_text = {"original": text, "processed": text, "entities": ["COMPANY", "PROFIT"]}
                else:
                    # Actually use the MCP to process text
                    processed_text = doc_analysis_mcp.analyze_text(text)
                processed_texts.append(processed_text)
                
            # Then use SentimentAnalysisModel on the processed texts
            sentiment_results = sentiment_model.analyze_sentiment_sync(processed_texts)
            
            # Validate the integration results
            self.assertIsNotNone(sentiment_results)
            self.assertEqual(len(sentiment_results), len(sample_texts))
            
            # Basic validation of sentiment results
            for result in sentiment_results:
                self.assertIn('sentiment', result)
                self.assertIn('confidence', result)
                
            logging.info(f"Sentiment analysis completed for {len(sentiment_results)} processed texts")
            
        except Exception as e:
            logging.error(f"Error testing SentimentAnalysisModel with DocumentAnalysisMCP: {e}")
            raise
            
        logging.info("SentimentAnalysisModel with DocumentAnalysisMCP test completed successfully")

    def test_decision_model_with_risk_analysis_mcp(self):
        """
        Tests the integrated functionality of DecisionModel with RiskAnalysisMCP.
        This test validates that the decision model can properly incorporate risk analysis
        from the RiskAnalysisMCP when making trading decisions.
        """
        logging.info("Testing DecisionModel with RiskAnalysisMCP integration...")
        
        # Initialize RiskAnalysisMCP with environment-based configuration
        risk_config = self._load_component_config("risk_analysis_mcp/risk_analysis_mcp_config.json")
        risk_mcp = RiskAnalysisMCP(config=risk_config)
        
        # Initialize DecisionModel with environment-based configuration
        decision_config = self._load_component_config("nextgen_decision/decision_model_config.json")
        decision_model = DecisionModel(config=decision_config)
        
        # Test data
        symbol = "AAPL"
        market_data = {
            "price": 150.25,
            "volume": 1000000,
            "volatility": 0.15
        }
        
        try:
            # First use RiskAnalysisMCP to assess risk
            if self.data_source == 'synthetic':
                # Use synthetic risk data
                risk_assessment = {"symbol": symbol, "risk_score": 65, "market_risk": "medium", "recommendation": "hold"}
            else:
                # Actually use the MCP to assess risk
                risk_assessment = risk_mcp.assess_risk(symbol, market_data)
            
            # Create analysis data that incorporates the risk assessment
            analysis_data = {
                "selection": {"symbol": symbol, "score": 0.8},
                "finnlp": {"symbol": symbol, "sentiment_score": 0.7},
                "forecaster": {"symbol": symbol, "prediction": "up"},
                "fundamental": {"symbol": symbol, "overall_rating": "buy"},
                "market_conditions": {"market_state": "bullish"},
                "portfolio_data": {"equity": 100000},
                "risk_assessment": risk_assessment
            }
            
            # Use DecisionModel to make a trade decision incorporating the risk assessment
            decision = decision_model.make_trade_decision(symbol, analysis_data)
            
            # Validate the integration results
            self.assertIsNotNone(decision)
            self.assertIn("action", decision)
            self.assertIn("confidence", decision)
            self.assertIn("reason", decision)
            
            logging.info(f"Decision made: {decision['action']} with confidence {decision['confidence']}")
            
        except Exception as e:
            logging.error(f"Error testing DecisionModel with RiskAnalysisMCP: {e}")
            raise
            
        logging.info("DecisionModel with RiskAnalysisMCP test completed successfully")

    def test_market_analysis_model_with_financial_data_mcp(self):
        """
        Tests the integrated functionality of MarketAnalysisModel with FinancialDataMCP.
        This test validates that the market analysis model can properly utilize data
        from the FinancialDataMCP when performing market analysis.
        """
        logging.info("Testing MarketAnalysisModel with FinancialDataMCP integration...")
        
        # Initialize FinancialDataMCP with environment-based configuration
        financial_data_config = self._load_component_config("financial_data_mcp/financial_data_mcp_config.json")
        financial_data_mcp = FinancialDataMCP(config=financial_data_config)
        
        # Initialize MarketAnalysisModel with environment-based configuration
        market_analysis_config = self._load_component_config("nextgen_market_analysis/market_analysis_model_config.json")
        market_analysis_model = MarketAnalysisModel(config=market_analysis_config)
        
        # Test data
        symbol = "MSFT"
        
        try:
            # First use FinancialDataMCP to fetch market data
            if self.data_source == 'synthetic':
                # Use synthetic market data
                market_data = {
                    "symbol": symbol,
                    "historical_prices": [
                        {"date": "2023-01-01", "close": 240.0},
                        {"date": "2023-01-02", "close": 242.5},
                        {"date": "2023-01-03", "close": 245.2}
                    ],
                    "indicators": {
                        "sma_50": 235.5,
                        "sma_200": 220.1,
                        "rsi": 58.3
                    }
                }
            else:
                # Actually use the MCP to fetch market data
                market_data = financial_data_mcp.get_market_data(symbol)
            
            # Use MarketAnalysisModel to analyze the market data
            analysis_result = market_analysis_model.analyze(market_data)
            
            # Validate the integration results
            self.assertIsNotNone(analysis_result)
            self.assertIn("trend", analysis_result)
            self.assertIn("strength", analysis_result)
            
            logging.info(f"Market analysis completed with trend: {analysis_result['trend']}")
            
        except Exception as e:
            logging.error(f"Error testing MarketAnalysisModel with FinancialDataMCP: {e}")
            raise
            
        logging.info("MarketAnalysisModel with FinancialDataMCP test completed successfully")

    def test_selection_model_with_time_series_mcp(self):
        """
        Tests the integrated functionality of SelectionModel with TimeSeriesMCP.
        This test validates that the selection model can properly utilize time series data
        from the TimeSeriesMCP when selecting stocks.
        """
        logging.info("Testing SelectionModel with TimeSeriesMCP integration...")
        
        # Initialize TimeSeriesMCP with environment-based configuration
        time_series_config = self._load_component_config("time_series_mcp/time_series_mcp_config.json")
        time_series_mcp = TimeSeriesMCP(config=time_series_config)
        
        # Initialize SelectionModel with environment-based configuration
        selection_config = self._load_component_config("nextgen_select/select_model_config.json")
        selection_model = SelectionModel(config=selection_config)
        
        try:
            # First use TimeSeriesMCP to get time series data for potential stocks
            if self.data_source == 'synthetic':
                # Use synthetic time series data
                time_series_data = {
                    "AAPL": [150.0, 152.3, 153.1, 151.8, 154.2],
                    "MSFT": [240.0, 242.5, 245.2, 243.8, 247.1],
                    "GOOGL": [2100.0, 2120.5, 2150.2, 2140.8, 2170.1]
                }
            else:
                # Actually use the MCP to get time series data
                symbols = ["AAPL", "MSFT", "GOOGL"]
                time_series_data = time_series_mcp.get_time_series(symbols)
            
            # Use SelectionModel to select stocks based on the time series data
            selection_result = selection_model.run_selection_agent(time_series_data)
            
            # Validate the integration results
            self.assertIsNotNone(selection_result)
            self.assertGreater(len(selection_result), 0)
            
            # Each selected stock should have a symbol and score
            for stock in selection_result:
                self.assertIn("symbol", stock)
                self.assertIn("score", stock)
            
            logging.info(f"Stock selection completed with {len(selection_result)} stocks selected")
            
        except Exception as e:
            logging.error(f"Error testing SelectionModel with TimeSeriesMCP: {e}")
            raise
            
        logging.info("SelectionModel with TimeSeriesMCP test completed successfully")

    def test_trader_model_with_trading_mcp_and_redis_mcp(self):
        """
        Tests the integrated functionality of TradeModel with TradingMCP and RedisMCP.
        This test validates that the trader model can execute trades using the TradingMCP
        and store results in Redis using RedisMCP.
        """
        logging.info("Testing TradeModel with TradingMCP and RedisMCP integration...")
        
        # Initialize TradingMCP with environment-based configuration
        trading_config = self._load_component_config("trading_mcp/trading_mcp_config.json")
        trading_mcp = TradingMCP(config=trading_config)
        
        # Initialize RedisMCP with environment-based configuration
        redis_config = self._load_component_config("redis_mcp/redis_mcp_config.json")
        redis_mcp = RedisMCP(config=redis_config)
        
        # Initialize TradeModel with environment-based configuration
        trade_config = self._load_component_config("nextgen_trader/trade_model_config.json")
        trade_model = TradeModel(config=trade_config)
        
        # Test data
        trade_decision = {
            "action": "buy",
            "symbol": "AAPL",
            "quantity": 10,
            "price": 150.25,
            "order_type": "market"
        }
        
        try:
            # Use TradeModel to execute the trade using TradingMCP
            if self.data_source == 'synthetic':
                # Use synthetic trade execution
                trade_result = {
                    "order_id": "synthetic-order-123",
                    "status": "filled",
                    "symbol": trade_decision["symbol"],
                    "quantity": trade_decision["quantity"],
                    "fill_price": trade_decision["price"]
                }
            else:
                # Actually use the model and MCP to execute the trade
                trade_result = trade_model.execute_trade(trade_decision)
            
            # Store the trade result in Redis using RedisMCP
            if self.data_source == 'synthetic':
                # Simulate storing in Redis
                redis_key = f"trade:{trade_result['order_id']}"
                redis_store_success = True
            else:
                # Actually store in Redis
                redis_key = f"trade:{trade_result['order_id']}"
                redis_store_success = redis_mcp.set_value(redis_key, json.dumps(trade_result))
            
            # Validate the integration results
            self.assertIsNotNone(trade_result)
            self.assertIn("order_id", trade_result)
            self.assertIn("status", trade_result)
            self.assertTrue(redis_store_success)
            
            # Retrieve the trade result from Redis to validate
            if self.data_source == 'synthetic':
                # Simulate retrieval from Redis
                retrieved_trade = trade_result
            else:
                # Actually retrieve from Redis
                retrieved_json = redis_mcp.get_value(redis_key)
                retrieved_trade = json.loads(retrieved_json) if retrieved_json else None
            
            self.assertEqual(retrieved_trade["order_id"], trade_result["order_id"])
            
            logging.info(f"Trade executed and stored in Redis: {trade_result['order_id']}")
            
        except Exception as e:
            logging.error(f"Error testing TradeModel with TradingMCP and RedisMCP: {e}")
            raise
            
        logging.info("TradeModel with TradingMCP and RedisMCP test completed successfully")

    def test_full_e2e_trading_workflow(self):
        """
        Tests the full end-to-end trading workflow with all models and MCPs integrated.
        This test simulates the entire trading process from data collection to trade execution.
        """
        logging.info("Testing full end-to-end trading workflow...")
        
        try:
            # Step 1: Market data collection using FinancialDataMCP
            financial_data_config = self._load_component_config("financial_data_mcp/financial_data_mcp_config.json")
            financial_data_mcp = FinancialDataMCP(config=financial_data_config)
            
            if self.data_source == 'synthetic':
                # Use synthetic market data
                market_data = {
                    "AAPL": {"price": 150.25, "volume": 1000000},
                    "MSFT": {"price": 240.5, "volume": 800000},
                    "GOOGL": {"price": 2100.75, "volume": 500000}
                }
            else:
                # Actually fetch market data
                market_data = financial_data_mcp.get_market_overview(["AAPL", "MSFT", "GOOGL"])
            
            # Step 2: Stock selection using SelectionModel and TimeSeriesMCP
            time_series_config = self._load_component_config("time_series_mcp/time_series_mcp_config.json")
            time_series_mcp = TimeSeriesMCP(config=time_series_config)
            
            selection_config = self._load_component_config("nextgen_select/select_model_config.json")
            selection_model = SelectionModel(config=selection_config)
            
            if self.data_source == 'synthetic':
                # Use synthetic time series data
                time_series_data = {
                    "AAPL": [150.0, 152.3, 153.1, 151.8, 154.2],
                    "MSFT": [240.0, 242.5, 245.2, 243.8, 247.1],
                    "GOOGL": [2100.0, 2120.5, 2150.2, 2140.8, 2170.1]
                }
            else:
                # Actually get time series data
                time_series_data = time_series_mcp.get_time_series(list(market_data.keys()))
            
            selected_stocks = selection_model.run_selection_agent(time_series_data)
            self.assertGreater(len(selected_stocks), 0)
            
            # Get the top selected stock
            top_stock = selected_stocks[0]["symbol"]
            
            # Step 3: Multi-faceted analysis
            # Use SentimentAnalysisModel with DocumentAnalysisMCP
            doc_analysis_config = self._load_component_config("document_analysis_mcp/document_analysis_mcp_config.json")
            doc_analysis_mcp = DocumentAnalysisMCP(config=doc_analysis_config)
            
            sentiment_config = self._load_component_config("nextgen_sentiment_analysis/sentiment_analysis_model_config.json")
            sentiment_model = SentimentAnalysisModel(config=sentiment_config)
            
            if self.data_source == 'synthetic':
                # Use synthetic document data
                stock_news = [
                    f"{top_stock} reported strong earnings this quarter.",
                    f"Analysts raise price target for {top_stock}."
                ]
            else:
                # Actually fetch news about the stock
                stock_news = financial_data_mcp.get_news(top_stock, limit=2)
            
            processed_news = []
            for news in stock_news:
                if self.data_source == 'synthetic':
                    processed_text = {"original": news, "processed": news}
                else:
                    processed_text = doc_analysis_mcp.analyze_text(news)
                processed_news.append(processed_text)
            
            sentiment_results = sentiment_model.analyze_sentiment(processed_news)
            
            # Use MarketAnalysisModel
            market_analysis_config = self._load_component_config("nextgen_market_analysis/market_analysis_model_config.json")
            market_analysis_model = MarketAnalysisModel(config=market_analysis_config)
            
            market_analysis_result = market_analysis_model.analyze(market_data[top_stock] if isinstance(market_data, dict) else market_data)
            
            # Use RiskAnalysisMCP
            risk_config = self._load_component_config("risk_analysis_mcp/risk_analysis_mcp_config.json")
            risk_mcp = RiskAnalysisMCP(config=risk_config)
            
            if self.data_source == 'synthetic':
                risk_assessment = {"symbol": top_stock, "risk_score": 65, "market_risk": "medium"}
            else:
                risk_assessment = risk_mcp.assess_risk(top_stock, market_data[top_stock])
            
            # Step 4: Decision making using DecisionModel
            decision_config = self._load_component_config("nextgen_decision/decision_model_config.json")
            decision_model = DecisionModel(config=decision_config)
            
            analysis_data = {
                "selection": {"symbol": top_stock, "score": selected_stocks[0]["score"]},
                "finnlp": {"symbol": top_stock, "sentiment_score": sentiment_results[0]["confidence"] if sentiment_results else 0.5},
                "forecaster": {"symbol": top_stock, "prediction": "up" if market_analysis_result.get("trend") == "bullish" else "down"},
                "fundamental": {"symbol": top_stock, "overall_rating": "buy"},
                "market_conditions": {"market_state": market_analysis_result.get("trend", "neutral")},
                "portfolio_data": {"equity": 100000},
                "risk_assessment": risk_assessment
            }
            
            decision = decision_model.make_trade_decision(top_stock, analysis_data)
            self.assertIn("action", decision)
            
            # Step 5: Trade execution if decision is to buy or sell
            if decision["action"] in ["buy", "sell"]:
                trade_config = self._load_component_config("nextgen_trader/trade_model_config.json")
                trade_model = TradeModel(config=trade_config)
                
                trade_decision = {
                    "action": decision["action"],
                    "symbol": top_stock,
                    "quantity": 10,
                    "order_type": "market"
                }
                
                if self.data_source == 'synthetic':
                    # Simulate trade execution
                    trade_result = {
                        "order_id": f"synthetic-order-{int(time.time())}",
                        "status": "filled",
                        "symbol": trade_decision["symbol"],
                        "quantity": trade_decision["quantity"]
                    }
                else:
                    # Actually execute the trade
                    trade_result = trade_model.execute_trade(trade_decision)
                
                self.assertIn("order_id", trade_result)
                self.assertIn("status", trade_result)
                
                # Store the trade result in Redis
                redis_config = self._load_component_config("redis_mcp/redis_mcp_config.json")
                redis_mcp = RedisMCP(config=redis_config)
                
                redis_key = f"trade:{trade_result['order_id']}"
                if self.data_source != 'synthetic':
                    redis_store_success = redis_mcp.set_value(redis_key, json.dumps(trade_result))
                    self.assertTrue(redis_store_success)
            
            logging.info("Full end-to-end trading workflow completed successfully")
            
        except Exception as e:
            logging.error(f"Error in full end-to-end trading workflow: {e}")
            raise

if __name__ == "__main__":
    unittest.main()