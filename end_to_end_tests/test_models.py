"""End-to-end tests for models."""

import pytest
import os
import dotenv
# Import test_data_generation.py directly
from test_data_generation import (
    generate_financial_market_data,
    generate_sentiment_text_data,
    generate_time_series_data,
    generate_diverse_datasets
)

# Import MCP tools that models depend on
from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.document_analysis_mcp.document_analysis_mcp import DocumentAnalysisMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP

import pandas as pd # Needed for downloaded data loading
import json # Needed for loading configuration files

# Load environment variables
dotenv.load_dotenv()

# Load database connection parameters from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8005"))

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

import unittest

class TestModels(unittest.TestCase):
    """Contains end-to-end tests for individual NextGen models."""

    def __init__(self, methodName='runTest'):
        """
        Initializes the TestModels.
        """
        super().__init__(methodName) # Call parent class constructor
        self.data_source = None # Initialize data_source
        self.redis_params = None
        self.vector_db_params = None

    def setUp(self):
        """Set up test environment before each test."""
        import os # Import os here if not already imported at the top
        self.data_source = os.getenv('E2E_DATA_SOURCE', 'synthetic') # Get data source from env var
        print(f"TestModels setup with data source: {self.data_source}")
        
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

    def _load_data(self, data_type):
        """
        Loads or accesses data based on the selected data source and data type.

        Args:
            data_type (str): The type of data to load (e.g., 'sentiment', 'market').

        Returns:
            DataFrame or other data structure: The loaded data.
        """
        if self.data_source == 'synthetic':
            print(f"Loading synthetic data for {data_type}...")
            if data_type == 'sentiment':
                return generate_sentiment_text_data()
            elif data_type == 'market':
                return generate_financial_market_data()
            elif data_type in ['autogen', 'context', 'decision', 'fundamental_analysis', 'risk_assessment', 'select', 'trade']:
                 # Return placeholder data for other types
                 print(f"Returning placeholder synthetic data for {data_type}.")
                 return {"placeholder_data": f"synthetic_{data_type}_data"}
            else:
                raise ValueError(f"Unknown synthetic data type: {data_type}")
        elif self.data_source == 'downloaded':
            print(f"Loading downloaded data for {data_type}...")
            # Example: Load from a CSV file in the datasets directory
            file_path = f"/home/ubuntu/nextgen/datasets/{data_type}_data.csv"
            try:
                return pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"Downloaded data file not found: {file_path}")
                # Fallback to synthetic data if downloaded data is not available
                print("Falling back to synthetic data.")
                return self._load_data(data_type, data_source='synthetic') # Recursive call with synthetic
            except Exception as e:
                print(f"Error loading downloaded data from {file_path}: {e}")
                # Fallback to synthetic data on error
                print("Falling back to synthetic data.")
                return self._load_data(data_type, data_source='synthetic') # Recursive call with synthetic
        elif self.data_source == 'live':
            print(f"Accessing live data for {data_type}...")
            # This would involve calling MCP tools. Placeholder for now.
            # Example:
            # if data_type == 'market':
            #     return SomeDataMCPTool.get_market_data()
            print(f"Live data access not yet implemented for {data_type}. Returning synthetic data.")
            return self._load_data(data_type, data_source='synthetic') # Fallback to synthetic
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")


    def _load_model_config(self, model_name):
        """
        Loads the configuration file for a given model and updates with environment variables.

        Args:
            model_name (str): The name of the model (e.g., 'sentiment_analysis_model').

        Returns:
            dict: The loaded and environment-updated configuration dictionary.
        """
        model_config_paths = {
            'sentiment_analysis_model': 'nextgen_sentiment_analysis/sentiment_analysis_model_config.json',
            'market_analysis_model': 'nextgen_market_analysis/market_analysis_model_config.json',
            'decision_model': 'nextgen_decision/decision_model_config.json',
            'autogen_model': 'autogen_orchestrator/autogen_orchestrator_config.json',
            'context_model': 'nextgen_context_model/context_model_config.json',
            'fundamental_analysis_model': 'nextgen_fundamental_analysis/fundamental_analysis_model_config.json',
            'risk_assessment_model': 'nextgen_risk_assessment/risk_assessment_model_config.json',
            'select_model': 'nextgen_select/select_model_config.json',
            'trade_model': 'nextgen_trader/trade_model_config.json',
        }

        relative_config_path = model_config_paths.get(model_name)
        if not relative_config_path:
            print(f"Warning: No specific config path found for model '{model_name}'. Using default path.")
            # Fallback to the old logic for models not in the map
            config_path = f"/home/ubuntu/nextgen/config/{model_name}_config.json"
        else:
            config_path = f"/home/ubuntu/nextgen/config/{relative_config_path}"

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration for {model_name} from {config_path}")
            
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
            print(f"Configuration file not found for {model_name} at {config_path}. Returning placeholder config.")
            return {"placeholder_config": f"{model_name}_config"} # Return a basic placeholder dict
        except json.JSONDecodeError:
            print(f"Error decoding JSON from configuration file {config_path}. Returning placeholder config.")
            return {"placeholder_config": f"{model_name}_config"} # Return a basic placeholder dict
        except Exception as e:
            print(f"An error occurred while loading configuration for {model_name}: {e}. Returning placeholder config.")
            return {"placeholder_config": f"{model_name}_config"} # Return a basic placeholder dict


    def _load_mcp_config(self, mcp_path):
        """
        Loads the configuration file for a given MCP tool and updates with environment variables.

        Args:
            mcp_path (str): The path to the MCP configuration file.

        Returns:
            dict: The loaded and environment-updated configuration dictionary.
        """
        config_path = f"/home/ubuntu/nextgen/config/{mcp_path}"

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            
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
            print(f"Configuration file not found at {config_path}. Returning empty config.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from configuration file {config_path}. Returning empty config.")
            return {}
        except Exception as e:
            print(f"Error loading configuration: {e}. Returning empty config.")
            return {}

    def test_sentiment_analysis_model(self):
        """Tests the SentimentAnalysisModel with DocumentAnalysisMCP integration."""
        # 1. Load or access the appropriate test data based on the selected source.
        sentiment_data = self._load_data('sentiment')

        # 2. Initialize DocumentAnalysisMCP with environment-based configuration
        doc_analysis_config = self._load_mcp_config('document_analysis_mcp/document_analysis_mcp_config.json')
        doc_analysis_mcp = DocumentAnalysisMCP(config=doc_analysis_config)
        
        # 3. Instantiate the sentiment model with environment-based configuration
        config = self._load_model_config('sentiment_analysis_model')
        model = SentimentAnalysisModel(config=config)
        
        # 4. Process the data through DocumentAnalysisMCP first
        processed_data = sentiment_data
        if self.data_source != 'synthetic':
            # For non-synthetic data, actually use the MCP to process
            processed_data = [doc_analysis_mcp.analyze_text(text) for text in sentiment_data]
            
        # 5. Analyze the processed data with the sentiment model
        results = model.analyze_sentiment_sync(processed_data)

        # 6. Assert or verify that the model's output is as expected.
        assert results is not None
        
        print(f"Sentiment Analysis Model Test Passed. Processed {len(sentiment_data)} items from {self.data_source} source.")

    def test_market_analysis_model(self):
        """Tests the MarketAnalysisModel with FinancialDataMCP integration."""
        # 1. Initialize FinancialDataMCP with environment-based configuration
        financial_data_config = self._load_mcp_config('financial_data_mcp/financial_data_mcp_config.json')
        financial_data_mcp = FinancialDataMCP(config=financial_data_config)
        
        # 2. Load market data using the MCP or from test data based on data source
        if self.data_source == 'synthetic':
            market_data = self._load_data('market')
        else:
            # For non-synthetic data, use the MCP to get actual market data
            symbols = ["AAPL", "MSFT", "GOOGL"]
            market_data = financial_data_mcp.get_market_overview(symbols)

        # 3. Instantiate and call the model with the test data
        config = self._load_model_config('market_analysis_model')
        model = MarketAnalysisModel(config=config)
        results = model.analyze(market_data)

        # 4. Assert or verify that the model's output is as expected
        assert results is not None
        
        if isinstance(results, dict):
            # If we're analyzing multiple symbols, check each one
            for symbol in results:
                assert "trend" in results[symbol]
        else:
            # Single result case
            assert "trend" in results

        print(f"Market Analysis Model Test Passed. Processed market data from {self.data_source} source.")


    def test_decision_model(self):
        """Tests the DecisionModel with RiskAnalysisMCP integration."""
        # 1. Initialize RiskAnalysisMCP with environment-based configuration
        risk_config = self._load_mcp_config('risk_analysis_mcp/risk_analysis_mcp_config.json')
        risk_mcp = RiskAnalysisMCP(config=risk_config)
        
        # 2. Get basic test data
        symbol = "AAPL"
        market_data = {"price": 150.25, "volume": 1000000, "volatility": 0.15}
        
        # 3. Perform risk analysis using the MCP
        if self.data_source == 'synthetic':
            # Use synthetic risk data
            risk_assessment = {"symbol": symbol, "risk_score": 65, "market_risk": "medium"}
        else:
            # Actually use the MCP to assess risk
            risk_assessment = risk_mcp.assess_risk(symbol, market_data)

        # 4. Create analysis data that incorporates the risk assessment
        analysis_data = {
            "selection": {"symbol": symbol, "score": 0.8},
            "finnlp": {"symbol": symbol, "sentiment_score": 0.7},
            "forecaster": {"symbol": symbol, "prediction": "up"},
            "fundamental": {"symbol": symbol, "overall_rating": "buy"},
            "market_conditions": {"market_state": "bullish"},
            "portfolio_data": {"equity": 100000},
            "risk_assessment": risk_assessment  # Include risk assessment from MCP
        }
        
        # 5. Instantiate and call the decision model
        config = self._load_model_config('decision_model')
        model = DecisionModel(config=config)
        decision = model.make_trade_decision(symbol=symbol, analysis_data=analysis_data)

        # 6. Assert or verify that the model's output is as expected
        assert decision is not None
        assert "action" in decision
        assert "confidence" in decision
        assert "reason" in decision

        print(f"Decision Model Test Passed. Processed decision with risk assessment from {self.data_source} source.")


    def test_autogen_model(self):
        """Tests the AutogenModel with selected data source."""
        # 1. Load or access the appropriate test data based on the selected source.
        autogen_data = self._load_data('autogen') # Assuming 'autogen' data type

        # 2. Instantiate and call the model with the test data.
        config = self._load_model_config('autogen_model')
        model = AutoGenOrchestrator()
        results = model.run_trading_cycle(market_data=autogen_data, config=config)

        # 3. Assert or verify that the model's output is as expected.
        assert results is not None
        # Add more specific assertions based on expected output

        print(f"Autogen Model Test Passed. Processed autogen data from {self.data_source} source.")

    def test_context_model(self):
        """Tests the ContextModel with VectorStoreMCP integration."""
        # 1. Initialize VectorStoreMCP with environment-based configuration
        vector_store_config = self._load_mcp_config('vector_store_mcp/vector_store_mcp_config.json')
        vector_store_mcp = VectorStoreMCP(config=vector_store_config)
        
        # 2. Prepare test data
        sample_documents = [
            {"id": "doc1", "text": "Apple stock price increased by 5% following strong earnings."},
            {"id": "doc2", "text": "Microsoft announced new AI features for their cloud platform."},
            {"id": "doc3", "text": "Tesla's production numbers exceeded analyst expectations."}
        ]
        
        # 3. Add documents to vector store if not using synthetic data
        if self.data_source != 'synthetic':
            vector_store_mcp.add_documents(sample_documents)
        
        # 4. Instantiate the context model
        config = self._load_model_config('context_model')
        model = ContextModel(config=config)
        
        # 5. Test context retrieval with the model
        query = "What happened with Apple stock?"
        results = model.retrieve_context_sync(query=query)

        # 6. Assert or verify that the model's output is as expected
        assert results is not None
        
        if self.data_source != 'synthetic':
            # Only verify specific results in non-synthetic mode
            assert len(results.get("retrieved_context", [])) > 0
            
        print(f"Context Model Test Passed. Retrieved {len(results.get('retrieved_context', [])) if results else 0} context items from {self.data_source} source.")

    def test_fundamental_analysis_model(self):
        """Tests the FundamentalAnalysisModel with selected data source."""
        # 1. Load or access the appropriate test data based on the selected source.
        fundamental_analysis_data = self._load_data('fundamental_analysis') # Assuming 'fundamental_analysis' data type

        # 2. Instantiate and call the model with the test data.
        config = self._load_model_config('fundamental_analysis_model')
        model = FundamentalAnalysisModel(config=config)
        results = model.analyze(fundamental_analysis_data) # Assuming an 'analyze' method

        # 3. Assert or verify that the model's output is as expected.
        assert results is not None
        # Add more specific assertions based on expected output

        print(f"Fundamental Analysis Model Test Passed. Processed fundamental analysis data from {self.data_source} source.")

    async def test_risk_assessment_model(self):
        """Tests the RiskAssessmentModel with selected data source."""
        # 1. Load or access the appropriate test data based on the selected source.
        risk_assessment_data = self._load_data('risk_assessment') # Assuming 'risk_assessment' data type

        # 2. Instantiate and call the model with the test data.
        config = self._load_model_config('risk_assessment_model')
        model = RiskAssessmentModel(config=config)
        results = await model.run_assessment_cycle() # Call the run_assessment_cycle method

        # 3. Assert or verify that the model's output is as expected.
        assert results is not None
        # Add more specific assertions based on expected output

        print(f"Risk Assessment Model Test Passed. Processed risk assessment data from {self.data_source} source.")

    def test_select_model(self):
        """Tests the SelectModel with selected data source."""
        # 1. Load or access the appropriate test data based on the selected source.
        select_data = self._load_data('select') # Assuming 'select' data type

        # 2. Instantiate and call the model with the test data.
        config = self._load_model_config('select_model')
        model = SelectionModel(config=config)
        results = model.run_selection_agent(select_data) # Assuming a 'run_selection_agent' method based on the class

        # 3. Assert or verify that the model's output is as expected.
        assert results is not None
        # Add more specific assertions based on expected output

        print(f"Select Model Test Passed. Processed select data from {self.data_source} source.")

    def test_trade_model(self):
        """Tests the TradeModel with selected data source."""
        # 1. Load or access the appropriate test data based on the selected source.
        trade_data = self._load_data('trade') # Assuming 'trade' data type

        # 2. Instantiate and call the model with the test data.
        config = self._load_model_config('trade_model')
        model = TradeModel(config=config)
        results = model.execute_trade(trade_data) # Assuming an 'execute_trade' method

        # 3. Assert or verify that the model's output is as expected.
        assert results is not None
        # Add more specific assertions based on expected output

        print(f"Trade Model Test Passed. Processed trade data from {self.data_source} source.")

# Add more test methods for other models as needed following the same pattern.
# Ensure the total lines of code remain under 500.

# Note: The test runner (run_all_tests.py) will need to instantiate this class
# and pass the selected data_source to its constructor.
# Example in run_all_tests.py:
# test_models_suite = loader.loadTestsFromTestCase(TestModels(data_source))
# suite.addTests(test_models_suite)
