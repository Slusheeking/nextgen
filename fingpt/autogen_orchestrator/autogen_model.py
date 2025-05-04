"""
AutoGen-Powered Orchestrator for FinGPT

This module implements the central orchestrator for the FinGPT system using Microsoft's AutoGen framework.
The orchestrator coordinates various specialized agents to perform financial analysis and trading decisions.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Callable, TypedDict

# Note: autogen is installed as package 'pyautogen' but imported as 'autogen'
import autogen  # type: ignore
from autogen import (  # type: ignore
    Agent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    ConversableAgent,
    register_function
    # FunctionCall is not available in autogen 0.9.0
)
from autogen.agentchat.contrib.capabilities.teachability import Teachability  # type: ignore

class AutoGenOrchestrator:
    """
    AutoGen-powered orchestrator for the FinGPT trading system.
    Acts as the central coordinator for all agent interactions.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the AutoGen orchestrator with configuration and agents.
        
        Args:
            config_path: Path to configuration JSON file (optional)
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.llm_config = self._get_llm_config()
        self.agents = {}
        self._setup_agents()
        self.group_chat = None
        self.chat_manager = None
        self._setup_group_chat()
        
        # Configuration for selection model integration
        self.max_candidates = self.config.get("selection", {}).get("max_candidates", 20)
        self.models = {}  # Will store model instances

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("fingpt.autogen_orchestrator")
        logger.setLevel(logging.INFO)
        
        # Add console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "llm": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", "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"),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
                        "api_version": None  # Explicitly set to None for OpenRouter
                    },
                    {
                        "model": "meta-llama/llama-3-70b-instruct",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", "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"),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
                        "api_version": None  # Explicitly set to None for OpenRouter
                    },
                    {
                        "model": "google/gemini-1.5-pro-latest",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", "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"),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
                        "api_version": None  # Explicitly set to None for OpenRouter
                    }
                ]
            },
            "agents": {
                "selection": {
                    "name": "SelectionAgent",
                    "system_message": "You are a stock selection specialist. Your role is to identify promising stocks for analysis based on market conditions, technical indicators, and news sentiment."
                },
                "data": {
                    "name": "DataAgent",
                    "system_message": "You are a financial data specialist. Your role is to gather, preprocess, and provide data needed for analysis and decision-making."
                },
                "finnlp": {
                    "name": "NLPAgent",
                    "system_message": "You are a financial sentiment analyst. Your role is to analyze news, social media, and other text sources to extract sentiment and insights."
                },
                "forecaster": {
                    "name": "ForecasterAgent",
                    "system_message": "You are a price forecasting specialist. Your role is to predict future price movements based on historical data and current market conditions."
                },
                "rag": {
                    "name": "RAGAgent",
                    "system_message": "You are a knowledge retrieval specialist. Your role is to provide relevant historical context and precedents to inform decision-making."
                },
                "execution": {
                    "name": "ExecutionAgent",
                    "system_message": "You are a trade execution specialist. Your role is to execute trading decisions efficiently while managing risk."
                },
                "monitoring": {
                    "name": "MonitoringAgent",
                    "system_message": "You are a system monitoring specialist. Your role is to track performance, detect anomalies, and ensure system health."
                }
            },
            "selection": {
                "max_candidates": 20,
                "min_price": 10.0,
                "max_price": 200.0,
                "min_volume": 500000,
                "min_relative_volume": 1.5,
                "max_spread_pct": 0.5
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            self.logger.warning(f"Config file not found at {config_path}. Using default configuration.")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Deep merge of default and user configs
            merged_config = default_config.copy()
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    merged_config[key] = {**default_config[key], **value}
                else:
                    merged_config[key] = value
                    
            return merged_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Using default configuration.")
            return default_config

    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen.
        
        Returns:
            LLM configuration dictionary
        """
        llm_section = self.config.get("llm", {})
        config_list = llm_section.get("config_list", [])
        
        # Log API configuration for debugging
        for i, config in enumerate(config_list):
            redacted_key = config.get("api_key", "")
            if redacted_key:
                redacted_key = redacted_key[:8] + "..." + redacted_key[-4:]
            self.logger.info(f"LLM Config {i+1}: model={config.get('model')}, api_type={config.get('api_type')}, api_key={redacted_key}")
            
        return {
            "config_list": config_list,
            "temperature": llm_section.get("temperature", 0.1),
            "timeout": llm_section.get("timeout", 600),
            "seed": 42  # Adding seed for reproducibility
        }

    def _setup_agents(self):
        """
        Initialize all AutoGen agents.
        """
        agent_configs = self.config.get("agents", {})
        
        # Create assistant agents for each role with AG2 configuration
        for role, config in agent_configs.items():
            self.agents[role] = AssistantAgent(
                name=config.get("name", f"{role.capitalize()} Agent"),
                system_message=config.get("system_message", "You are a financial AI assistant."),
                llm_config=self.llm_config,
                description=config.get("description", f"A specialized agent for {role} tasks in financial analysis")
            )
            
        # Create a human proxy agent for testing and interactivity with AG2 configuration
        self.agents["human_proxy"] = UserProxyAgent(
            name="HumanProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"use_docker": False},  # Disable Docker since it's not available
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that."
        )
        
        self.logger.info(f"Created {len(self.agents)} agents: {list(self.agents.keys())}")

    def _setup_group_chat(self):
        """
        Set up group chat for multi-agent collaboration using AG2 patterns.
        """
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin",  # AG2 supports different speaker selection methods
            allow_repeat_speaker=False
        )
        
        self.chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            system_message="You are a group chat manager for a financial trading system. Your job is to facilitate productive discussion between specialized agents."
        )
        
        self.logger.info("Group chat and manager initialized")

    def run_trading_cycle(self, market_data: Optional[Dict[str, Any]] = None, mock: bool = False, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a complete AutoGen-orchestrated trading workflow.
        
        Args:
            market_data: Optional market data to provide context
            mock: If True, run in test mode without making actual API calls
            config: Optional configuration for models
            
        Returns:
            Results of the trading cycle including decisions
        """
        if mock:
            self.logger.warning("Running in MOCK mode - no API calls will be made")
            # Return mock result
            return {
                "chat_result": "This is a mock response since mock=True was specified. No API calls were made.",
                "decisions": [],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
        self.logger.info("Running with REAL LLM API calls to OpenRouter")
        self.logger.info("Starting new trading cycle")
        
        # Connect to models if not already connected
        if not hasattr(self, 'models') or not self.models:
            self.connect_to_models(config)
        
        # Run the selection model to get candidates if available
        selected_candidates = []
        market_context = {}
        selection_data = {}
        
        if hasattr(self, 'models') and 'selection' in self.models:
            self.logger.info("Getting data from SelectionModel")
            try:
                # Get comprehensive selection data
                selection_data = self.models['selection'].get_selection_data()
                self.logger.info(f"Retrieved selection data with {len(selection_data.get('candidates', []))} candidates")
                
                # Extract market context and candidates
                market_context = selection_data.get('market_context', {})
                selected_candidates = selection_data.get('top_candidates', [])
                
                # If no candidates in top_candidates, try all candidates
                if not selected_candidates:
                    selected_candidates = selection_data.get('candidates', [])
                
                # If still no candidates, run a new selection cycle
                if not selected_candidates:
                    self.logger.info("No candidates found in Redis, running new selection cycle")
                    selected_candidates = self.models['selection'].run_selection_cycle()
                    self.logger.info(f"SelectionModel identified {len(selected_candidates)} candidates")
            except Exception as e:
                self.logger.error(f"Error getting selection data: {e}")
                self.logger.exception("Exception details:")
        
        # Reset the group chat for a new trading cycle
        self.group_chat.reset()
        
        # Create initial prompt with market context and selection results
        market_context_str = ""
        
        # Use market context from SelectionModel if available, otherwise use provided market_data
        if market_context:
            market_context_str = f"""
            Current market context:
            - S&P 500: {market_context.get('spy_price', market_data.get('sp500', 'N/A'))}
            - VIX: {market_context.get('vix', market_data.get('vix', 'N/A'))}
            - Market open: {market_context.get('market_open', 'N/A')}
            - Buying power: ${market_context.get('buying_power', 'N/A'):,.2f}
            - Portfolio value: ${market_context.get('portfolio_value', 'N/A'):,.2f}
            """
        elif market_data:
            market_context_str = f"""
            Current market context:
            - S&P 500: {market_data.get('sp500', 'N/A')}
            - VIX: {market_data.get('vix', 'N/A')}
            - 10-Year Treasury: {market_data.get('treasury_10y', 'N/A')}
            - Market sentiment: {market_data.get('market_sentiment', 'N/A')}
            """
        
        # Add selection results to the prompt
        selection_context = ""
        if selected_candidates:
            selection_context = "\nSelected candidates from the SelectionModel:\n"
            for i, candidate in enumerate(selected_candidates[:5]):  # Show top 5
                selection_context += f"- {candidate.get('ticker')}: Score {candidate.get('score', 0):.2f}, "
                selection_context += f"RSI {candidate.get('rsi', 0):.2f}, "
                selection_context += f"Relative Volume {candidate.get('relative_volume', 0):.2f}x\n"
            
            if len(selected_candidates) > 5:
                selection_context += f"- Plus {len(selected_candidates) - 5} more candidates...\n"
        
        initial_prompt = f"""
        {market_context_str}
        
        {selection_context}
        
        We are starting a new trading cycle. Please follow this structured workflow:
        
        1. First, the Selection Agent should review the candidates already identified by the SelectionModel and select 3-5 of the most promising stocks for further analysis.
           - Consider technical indicators (RSI, moving averages, volatility)
           - Consider unusual activity signals
           - Consider relative volume and liquidity
           - Provide detailed reasoning for each selection
        
        2. For each selected stock, the Data Agent should gather relevant financial data, including:
           - Recent price movements
           - Trading volumes
           - Key financial metrics
           - Support and resistance levels
        
        3. The NLP Agent should analyze news sentiment for each selected stock.
        
        4. The Forecaster Agent should provide price predictions for each stock.
        
        5. The RAG Agent should provide relevant historical context about similar market conditions or company events.
        
        6. Based on all inputs, collectively decide on trading actions (buy, sell, hold) with specific quantities and reasoning.
        
        7. The Execution Agent should outline how these trades would be executed.
        
        8. The Monitoring Agent should suggest metrics to track for these positions.
        
        Selection Agent, please start by reviewing the candidates and selecting the most promising stocks for analysis.
        """
        
        # Run the group chat with the initial prompt
        self.logger.info("Initiating group chat for trading workflow")
        self.logger.info("About to call chat_manager.run() with initial prompt...")
        
        try:
            # Use a more direct approach - first message to selection agent
            self.logger.info("Starting with selection agent...")
            selection_agent = self.agents.get("selection")
            human_proxy = self.agents.get("human_proxy")
            
            if selection_agent and human_proxy:
                # Start with a direct message to initiate the workflow
                self.logger.info("Sending direct message to selection agent...")
                human_proxy.initiate_chat(
                    selection_agent,
                    message=initial_prompt
                )
                
                # Get the chat history
                messages = selection_agent.chat_history[human_proxy]
                combined_result = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                self.logger.info(f"Got {len(messages)} messages in chat history")
                
                response = combined_result
            else:
                # Fallback to standard group chat if agents not found
                self.logger.info("Falling back to standard group chat...")
                response = self.chat_manager.run(initial_prompt)
                self.logger.info(f"Response type: {type(response)}")
                
        except Exception as e:
            self.logger.error(f"Error in chat execution: {str(e)}")
            self.logger.exception("Exception details:")
            return {"chat_result": f"Error: {str(e)}",
                   "decisions": [],
                   "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        
        # Extract messages from the response
        # In newer AutoGen versions, we need to handle RunResponse specially
        try:
            # Try to access the messages property or attribute
            if hasattr(response, 'messages'):
                messages = response.messages
                # Concatenate messages into a single string
                chat_result = "\n\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                                          for msg in messages if isinstance(msg, dict)])
            elif hasattr(response, 'chat_history'):
                chat_result = "\n\n".join([f"{msg[0]}: {msg[1]}" for msg in response.chat_history])
            else:
                # Fallback: try different properties or convert to string
                chat_result = str(response)
                self.logger.warning("Using fallback string conversion for RunResponse")
        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            chat_result = str(response)
        
        # Extract and process results
        decisions = self._extract_trading_decisions(chat_result)
        
        # Log the outcomes
        self.logger.info(f"Trading cycle completed with {len(decisions)} decisions")
        
        return {
            "chat_result": chat_result,
            "decisions": decisions,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
    def _extract_trading_decisions(self, chat_result: str) -> List[Dict[str, Any]]:
        """
        Extract structured trading decisions from the group chat result.
        
        Args:
            chat_result: Text output from the group chat
            
        Returns:
            List of trading decisions
        """
        # This is a simplified implementation
        # In a production system, this would use regex or LLM-based extraction
        # to parse the decisions from the chat output
        
        decisions = []
        
        # Very basic parsing - in production would be more sophisticated
        if chat_result and "DECISION:" in chat_result:
            decision_sections = chat_result.split("DECISION:")[1:]
            
            for section in decision_sections:
                lines = section.split("\n")
                ticker = None
                action = None
                quantity = None
                reason = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Ticker:"):
                        ticker = line.replace("Ticker:", "").strip()
                    elif line.startswith("Action:"):
                        action = line.replace("Action:", "").strip().lower()
                    elif line.startswith("Quantity:"):
                        try:
                            quantity = int(line.replace("Quantity:", "").strip())
                        except ValueError:
                            quantity = 0
                    elif line.startswith("Reason:"):
                        reason = line.replace("Reason:", "").strip()
                        
                if ticker and action:
                    decisions.append({
                        "ticker": ticker,
                        "action": action,
                        "quantity": quantity,
                        "reason": reason
                    })
        
        return decisions

    def connect_to_models(self, config=None):
        """
        Connect to FinGPT model implementations.
        
        This method integrates with the actual FinGPT model implementations,
        allowing the agents to call the model methods.
        
        Args:
            config: Optional configuration dictionary for the models
        """
        self.models = {}
        
        try:
            from fingpt.nextgen_select.select_model import SelectionModel
            from fingpt.fingpt_finnlp.finnlp_model import FinNLPModel
            from fingpt.fingpt_forcaster.forcaster_model import ForcasterModel
            from fingpt.fingpt_rag.rag_model import RAGModel
            from fingpt.nextgen_trader.trade_model import ExecutionModel
            from fingpt.nextgen_select.select_model import PositionMonitorModel
            from fingpt.fingpt_bench.bench_model import BenchModel
            from fingpt.fingpt_lora.lora_model import LoraModel
            
            self.logger.info("Successfully imported FinGPT models")
            
            # Initialize models with configuration
            model_config = config or {}
            
            # Initialize the Selection Model with proper configuration
            selection_config = model_config.get("selection_config", {})
            
            # Ensure selection_config has necessary MCP configurations
            if "alpaca_config" not in selection_config and "alpaca_config" in model_config:
                selection_config["alpaca_config"] = model_config.get("alpaca_config")
            if "redis_config" not in selection_config and "redis_config" in model_config:
                selection_config["redis_config"] = model_config.get("redis_config")
            if "polygon_rest_config" not in selection_config and "polygon_rest_config" in model_config:
                selection_config["polygon_rest_config"] = model_config.get("polygon_rest_config")
            if "polygon_ws_config" not in selection_config and "polygon_ws_config" in model_config:
                selection_config["polygon_ws_config"] = model_config.get("polygon_ws_config")
            if "unusual_whales_config" not in selection_config and "unusual_whales_config" in model_config:
                selection_config["unusual_whales_config"] = model_config.get("unusual_whales_config")
            
            # Pass the LLM config to the selection model if not already set
            if "llm_config" not in selection_config:
                selection_config["llm_config"] = self.llm_config
                
            self.models["selection"] = SelectionModel(selection_config)
            self.logger.info("Initialized SelectionModel")
            
            # Initialize other models as they become available
            # self.models["finnlp"] = FinNLPModel(model_config.get("finnlp_config", {}))
            # self.models["forecaster"] = ForcasterModel(model_config.get("forecaster_config", {}))
            # self.models["rag"] = RAGModel(model_config.get("rag_config", {}))
            # self.models["execution"] = ExecutionModel(model_config.get("execution_config", {}))
            # self.models["position_monitor"] = PositionMonitorModel(model_config.get("position_monitor_config", {}))
            # self.models["bench"] = BenchModel(model_config.get("bench_config", {}))
            # self.models["lora"] = LoraModel(model_config.get("lora_config", {}))
            
            # Register model methods with the agents
            self._register_model_methods()
            
        except ImportError as e:
            self.logger.warning(f"Could not import FinGPT models: {e}. Running in standalone mode.")
    
    def _register_model_methods(self):
        """
        Register model methods with the appropriate agents using AG2 patterns.
        
        This connects the specialized model implementations with the AutoGen agents,
        allowing the agents to call model methods directly using the new function registration pattern.
        """
        # Get the user proxy agent
        user_proxy = self.agents.get("human_proxy")
        if not user_proxy:
            self.logger.warning("Could not register model methods: user proxy agent not found")
            return
            
        # Register SelectionModel methods
        if "selection" in self.models:
            self.logger.info("Registering SelectionModel methods with user proxy agent")
            
            selection_model = self.models["selection"]
            
            # Register functions using AG2's register_function decorator
            # Core selection methods
            @register_function(
                name="run_selection_cycle",
                description="Run a complete selection cycle to identify trading candidates",
                return_type=List[Dict[str, Any]]
            )
            def run_selection_cycle() -> List[Dict[str, Any]]:
                return selection_model.run_selection_cycle()
            
            @register_function(
                name="get_selection_data",
                description="Get comprehensive selection data including market context and candidates",
                return_type=Dict[str, Any]
            )
            def get_selection_data() -> Dict[str, Any]:
                return selection_model.get_selection_data()
            
            # Data gathering methods
            @register_function(
                name="get_market_data",
                description="Get initial universe of stocks",
                return_type=List[Dict[str, Any]]
            )
            def get_market_data() -> List[Dict[str, Any]]:
                return selection_model.get_market_data()
            
            @register_function(
                name="get_technical_indicators",
                description="Calculate technical indicators for the stocks",
                parameters={
                    "stocks": {
                        "type": "array",
                        "description": "List of stock dictionaries"
                    }
                },
                return_type=List[Dict[str, Any]]
            )
            def get_technical_indicators(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return selection_model.get_technical_indicators(stocks)
            
            @register_function(
                name="get_unusual_activity",
                description="Check for unusual activity in the stocks",
                parameters={
                    "stocks": {
                        "type": "array",
                        "description": "List of stock dictionaries"
                    }
                },
                return_type=List[Dict[str, Any]]
            )
            def get_unusual_activity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return selection_model.get_unusual_activity(stocks)
            
            # Filtering and scoring methods
            @register_function(
                name="filter_by_liquidity",
                description="Apply liquidity filters to the universe of stocks",
                parameters={
                    "stocks": {
                        "type": "array",
                        "description": "List of stock dictionaries"
                    }
                },
                return_type=List[Dict[str, Any]]
            )
            def filter_by_liquidity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return selection_model.filter_by_liquidity(stocks)
            
            @register_function(
                name="score_candidates",
                description="Score and rank the candidate stocks",
                parameters={
                    "stocks": {
                        "type": "array",
                        "description": "List of stock dictionaries"
                    }
                },
                return_type=List[Dict[str, Any]]
            )
            def score_candidates(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return selection_model.score_candidates(stocks)
            
            # Storage and retrieval methods
            @register_function(
                name="store_candidates",
                description="Store candidates in Redis",
                parameters={
                    "candidates": {
                        "type": "array",
                        "description": "List of candidate dictionaries"
                    }
                },
                return_type=bool
            )
            def store_candidates(candidates: List[Dict[str, Any]]) -> bool:
                return selection_model.store_candidates(candidates)
            
            @register_function(
                name="get_candidates",
                description="Get the current list of candidates from Redis",
                return_type=List[Dict[str, Any]]
            )
            def get_candidates() -> List[Dict[str, Any]]:
                return selection_model.get_candidates()
            
            @register_function(
                name="get_top_candidates",
                description="Get the top N candidates by score",
                parameters={
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of candidates to return",
                        "default": 10
                    }
                },
                return_type=List[Dict[str, Any]]
            )
            def get_top_candidates(limit: int = 10) -> List[Dict[str, Any]]:
                return selection_model.get_top_candidates(limit)
            
            @register_function(
                name="get_candidate",
                description="Get a specific candidate by symbol",
                parameters={
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    }
                },
                return_type=Optional[Dict[str, Any]]
            )
            def get_candidate(symbol: str) -> Optional[Dict[str, Any]]:
                return selection_model.get_candidate(symbol)
            
            @register_function(
                name="get_market_context",
                description="Get current market context",
                return_type=Dict[str, Any]
            )
            def get_market_context() -> Dict[str, Any]:
                return selection_model._get_market_context()
            
            # Register all functions with the user proxy agent using AG2 pattern
            user_proxy.register_function(run_selection_cycle)
            user_proxy.register_function(get_selection_data)
            user_proxy.register_function(get_market_data)
            user_proxy.register_function(get_technical_indicators)
            user_proxy.register_function(get_unusual_activity)
            user_proxy.register_function(filter_by_liquidity)
            user_proxy.register_function(score_candidates)
            user_proxy.register_function(store_candidates)
            user_proxy.register_function(get_candidates)
            user_proxy.register_function(get_top_candidates)
            user_proxy.register_function(get_candidate)
            user_proxy.register_function(get_market_context)
            
            self.logger.info("Registered SelectionModel methods with user proxy agent using AG2 patterns")
            
            # Update the selection agent's system message to include available functions
            selection_agent = self.agents.get("selection")
            if selection_agent:
                # Enhance the system message with available functions
                enhanced_message = selection_agent.system_message + "\n\nYou have access to the following functions:\n"
                
                # List of registered functions
                function_names = [
                    "run_selection_cycle", "get_selection_data", "get_market_data",
                    "get_technical_indicators", "get_unusual_activity", "filter_by_liquidity",
                    "score_candidates", "store_candidates", "get_candidates",
                    "get_top_candidates", "get_candidate", "get_market_context"
                ]
                
                for func_name in function_names:
                    enhanced_message += f"- {func_name}()\n"
                enhanced_message += "\nUse these functions to access market data, analyze stocks, and make selection decisions."
                
                # Update the agent's system message
                selection_agent.update_system_message(enhanced_message)
                self.logger.info("Updated selection agent's system message with available functions")
                
            # Register MCP tool access functions
            self._register_mcp_tool_access(user_proxy)
            
    def _register_mcp_tool_access(self, user_proxy: UserProxyAgent):
        """
        Register MCP tool access functions with the user proxy agent.
        
        Args:
            user_proxy: The user proxy agent to register functions with
        """
        self.logger.info("Registering MCP tool access functions")
        
        @register_function(
            name="use_mcp_tool",
            description="Use a tool provided by an MCP server",
            parameters={
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server providing the tool"
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool"
                }
            },
            return_type=Any
        )
        def use_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
            """
            Use a tool provided by an MCP server.
            
            Args:
                server_name: Name of the MCP server providing the tool
                tool_name: Name of the tool to execute
                arguments: Arguments for the tool
                
            Returns:
                Result of the tool execution
            """
            # Get the MCP server instance
            if "selection" in self.models:
                selection_model = self.models["selection"]
                
                # Access MCP servers from the selection model
                if server_name == "alpaca":
                    mcp_server = selection_model.alpaca_mcp
                elif server_name == "redis":
                    mcp_server = selection_model.redis_mcp
                elif server_name == "polygon_rest":
                    mcp_server = selection_model.polygon_rest_mcp
                elif server_name == "polygon_ws":
                    mcp_server = selection_model.polygon_ws_mcp
                elif server_name == "unusual_whales":
                    mcp_server = selection_model.unusual_whales_mcp
                else:
                    return {"error": f"MCP server not found: {server_name}"}
                
                # Call the tool
                try:
                    return mcp_server.call_tool(tool_name, arguments)
                except Exception as e:
                    return {"error": str(e)}
            else:
                return {"error": "Selection model not initialized"}
        
        @register_function(
            name="list_mcp_tools",
            description="List all available tools on an MCP server",
            parameters={
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server"
                }
            },
            return_type=List[Dict[str, str]]
        )
        def list_mcp_tools(server_name: str) -> List[Dict[str, str]]:
            """
            List all available tools on an MCP server.
            
            Args:
                server_name: Name of the MCP server
                
            Returns:
                List of tool information dictionaries
            """
            # Get the MCP server instance
            if "selection" in self.models:
                selection_model = self.models["selection"]
                
                # Access MCP servers from the selection model
                if server_name == "alpaca":
                    mcp_server = selection_model.alpaca_mcp
                elif server_name == "redis":
                    mcp_server = selection_model.redis_mcp
                elif server_name == "polygon_rest":
                    mcp_server = selection_model.polygon_rest_mcp
                elif server_name == "polygon_ws":
                    mcp_server = selection_model.polygon_ws_mcp
                elif server_name == "unusual_whales":
                    mcp_server = selection_model.unusual_whales_mcp
                else:
                    return [{"error": f"MCP server not found: {server_name}"}]
                
                # List the tools
                try:
                    return mcp_server.list_tools()
                except Exception as e:
                    return [{"error": str(e)}]
            else:
                return [{"error": "Selection model not initialized"}]
        
        # Register the MCP tool access functions
        user_proxy.register_function(use_mcp_tool)
        user_proxy.register_function(list_mcp_tools)
        
        self.logger.info("Registered MCP tool access functions with user proxy agent")

    def execute_trades(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the trading decisions using the execution model.
        
        Args:
            decisions: List of trading decisions
            
        Returns:
            Execution results
        """
        # This would connect to the execution model and broker API
        # For now, it's just a placeholder
        self.logger.info(f"Would execute {len(decisions)} trades")
        
        execution_results = {
            "total": len(decisions),
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        # In a real implementation, this would call the execution model
        # from fingpt.fingpt_execution.execution_model import ExecutionModel
        # execution_model = ExecutionModel()
        # for decision in decisions:
        #     result = execution_model.execute_trade(decision)
        #     execution_results["details"].append(result)
        #     if result.get("success"):
        #         execution_results["success"] += 1
        #     else:
        #         execution_results["failed"] += 1
        
        return execution_results


def create_orchestrator(config_path: Optional[str] = None, model_config: Optional[Dict[str, Any]] = None) -> AutoGenOrchestrator:
    """
    Factory function to create and initialize an AutoGenOrchestrator.
    
    Args:
        config_path: Optional path to configuration file
        model_config: Optional configuration for models
        
    Returns:
        Initialized AutoGenOrchestrator
    """
    orchestrator = AutoGenOrchestrator(config_path)
    
    # Connect to models if configuration is provided
    if model_config:
        orchestrator.connect_to_models(model_config)
        
    return orchestrator


if __name__ == "__main__":
    # Simple test usage
    print("Initializing AutoGen orchestrator with OpenRouter models...")
    print("Available models: anthropic/claude-3-opus, meta-llama/llama-3-70b, google/gemini-1.5-pro")
    
    orchestrator = create_orchestrator()
    
    # Run a test trading cycle with some market data
    sample_market_data = {
        "sp500": 5024.35,
        "vix": 14.87,
        "treasury_10y": 4.12,
        "market_sentiment": "mixed"
    }
    
    print("Running trading cycle with sample market data...")
    print(f"Using OpenRouter API with key: sk-or-v1-0b39...d885")
    
    # Sample model configuration with MCP configurations
    model_config = {
        # MCP configurations that will be shared with models
        "alpaca_config": {
            "api_key": os.environ.get("ALPACA_API_KEY", ""),
            "api_secret": os.environ.get("ALPACA_API_SECRET", ""),
            "paper": True  # Use paper trading
        },
        "redis_config": {
            "host": os.environ.get("REDIS_HOST", "localhost"),
            "port": int(os.environ.get("REDIS_PORT", 6379)),
            "db": int(os.environ.get("REDIS_DB", 0))
        },
        "polygon_rest_config": {
            "api_key": os.environ.get("POLYGON_API_KEY", "")
        },
        "polygon_ws_config": {
            "api_key": os.environ.get("POLYGON_API_KEY", "")
        },
        "unusual_whales_config": {
            "api_key": os.environ.get("UNUSUAL_WHALES_API_KEY", "")
        },
        
        # Selection model specific configuration
        "selection_config": {
            "max_candidates": 15,
            "min_price": 10.0,
            "max_price": 200.0,
            "min_volume": 500000,
            "min_relative_volume": 1.5,
            "max_spread_pct": 0.5,
            "redis_prefix": "selection:",
            "llm_config": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None
                    }
                ]
            }
        }
    }
    
    # Run the trading cycle with the configuration
    result = orchestrator.run_trading_cycle(sample_market_data, config=model_config)
    print(f"Trading cycle completed with {len(result.get('decisions', []))} decisions")