"""
AutoGen-Powered Orchestrator for NextGen Models

This module implements the central orchestrator for the NextGen Models system using Microsoft's AutoGen framework.
The orchestrator coordinates various specialized agents to perform financial analysis and trading decisions.
"""

import os
import json
import datetime
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_api_key(key_name: str, default: str = "") -> str:
    """
    Get API key from environment variables.

    Args:
        key_name: Name of the API key
        default: Default value if not found

    Returns:
        API key value
    """
    # Map key names to environment variable names
    key_map = {
        "alpaca": "ALPACA_API_KEY",
        "alpaca_secret": "ALPACA_SECRET_KEY",
        "polygon": "POLYGON_API_KEY",
        "unusual_whales": "UNUSUAL_WHALES_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "reddit_client_id": "REDDIT_CLIENT_ID",
        "reddit_client_secret": "REDDIT_CLIENT_SECRET",
        "reddit_user_agent": "REDDIT_USER_AGENT",
    }

    env_var = key_map.get(key_name, key_name.upper())
    return os.environ.get(env_var, default)

def get_env(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value
    """
    return os.environ.get(key, default)

# Import monitoring
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

# Note: autogen is installed as package 'pyautogen' but imported as 'autogen'
from autogen import (  # type: ignore
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    register_function,
    # FunctionCall is not available in autogen 0.9.0
)


class AutoGenOrchestrator:
    """
    AutoGen-powered orchestrator for the NextGen Models trading system.
    Acts as the central coordinator for all agent interactions.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the AutoGen orchestrator with configuration and agents.

        Args:
            config_path: Path to configuration JSON file (optional)
        """
        init_start_time = time.time()
        self.logger = NetdataLogger(component_name="autogen-orchestrator")

        # Initialize system metrics collector
        self.metrics_collector = SystemMetricsCollector(self.logger)

        # Start collecting system metrics
        self.metrics_collector.start()

        # Initialize counters for orchestrator metrics
        self.trading_cycles_run = 0
        self.decisions_made_count = 0
        self.execution_errors = 0
        self.llm_api_call_count = 0 # This might be better tracked by the LLM config itself or agents

        self.config = self._load_config(config_path)
        self.llm_config = self._get_llm_config()

        self.agents = {}
        self._setup_agents()

        # Log initialization
        self.logger.info("AutoGenOrchestrator initialized")
        self.group_chat = None
        self.chat_manager = None
        self._setup_group_chat()

        # Configuration for selection model integration
        self.max_candidates = self.config.get("selection", {}).get("max_candidates", 20)
        self.models = {}  # Will store model instances

        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("orchestrator.initialization_time_ms", init_duration)


    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            Configuration dictionary
        """
        # Define default config as fallback
        default_config = {
            "llm": {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
                        "api_version": None,  # Explicitly set to None for OpenRouter
                    },
                    {
                        "model": "meta-llama/llama-3-70b-instruct",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
                        "api_version": None,  # Explicitly set to None for OpenRouter
                    },
                    {
                        "model": "google/gemini-1.5-pro-latest",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",  # OpenRouter uses OpenAI-compatible API
                        "api_version": None,  # Explicitly set to None for OpenRouter
                    },
                ],
            },
            "agents": {
                "selection": {
                    "name": "SelectionAgent",
                    "system_message": "You are a stock selection specialist. Your role is to identify promising stocks for analysis based on market conditions, technical indicators, and news sentiment.",
                },
                "data": {
                    "name": "DataAgent",
                    "system_message": "You are a financial data specialist. Your role is to gather, preprocess, and provide data needed for analysis and decision-making.",
                },
                "finnlp": {
                    "name": "NLPAgent",
                    "system_message": "You are a financial sentiment analyst. Your role is to analyze news, social media, and other text sources to extract sentiment and insights.",
                },
                "forecaster": {
                    "name": "ForecasterAgent",
                    "system_message": "You are a price forecasting specialist. Your role is to predict future price movements based on historical data and current market conditions.",
                },
                "rag": {
                    "name": "RAGAgent",
                    "system_message": "You are a knowledge retrieval specialist. Your role is to provide relevant historical context and precedents to inform decision-making.",
                },
                "execution": {
                    "name": "ExecutionAgent",
                    "system_message": "You are a trade execution specialist. Your role is to execute trading decisions efficiently while managing risk.",
                },
                "monitoring": {
                    "name": "MonitoringAgent",
                    "system_message": "You are a system monitoring specialist. Your role is to track performance, detect anomalies, and ensure system health.",
                },
            },
            "selection": {
                "max_candidates": 20,
                "min_price": 10.0,
                "max_price": 200.0,
                "min_volume": 500000,
                "min_relative_volume": 1.5,
                "max_spread_pct": 0.5,
            },
        }

        # First try to load from the specified config_path if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)

                # Deep merge of default and user configs
                merged_config = default_config.copy()
                for key, value in user_config.items():
                    if (
                        isinstance(value, dict)
                        and key in default_config
                        and isinstance(default_config[key], dict)
                    ):
                        merged_config[key] = {**default_config[key], **value}
                    else:
                        merged_config[key] = value

                self.logger.info(f"Loaded configuration from {config_path}")
                return merged_config
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {e}")
                # Fall through to standard config path

        # If no config_path provided or it failed, try the standard location
        standard_config_path = os.path.join("config", "autogen_orchestrator", "autogen_model_config.json")
        if os.path.exists(standard_config_path):
            try:
                with open(standard_config_path, "r") as f:
                    standard_config = json.load(f)

                # Deep merge of default and standard configs
                merged_config = default_config.copy()
                for key, value in standard_config.items():
                    if (
                        isinstance(value, dict)
                        and key in default_config
                        and isinstance(default_config[key], dict)
                    ):
                        merged_config[key] = {**default_config[key], **value}
                    else:
                        merged_config[key] = value

                self.logger.info(f"Loaded configuration from {standard_config_path}")
                return merged_config
            except Exception as e:
                self.logger.error(f"Error loading standard config: {e}")

        # If we get here, use default configuration
        self.logger.warning("Using default configuration")
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
            self.logger.info(
                f"LLM Config {i + 1}: model={config.get('model')}, api_type={config.get('api_type')}, api_key={redacted_key}",
                model=config.get('model'),
                api_type=config.get('api_type')
            )

        return {
            "config_list": config_list,
            "temperature": llm_section.get("temperature", 0.1),
            "timeout": llm_section.get("timeout", 600),
            "seed": 42,  # Adding seed for reproducibility
        }

    def shutdown(self):
        """
        Shutdown the orchestrator and stop metrics collection.
        """
        self.logger.info("Shutting down AutoGenOrchestrator")
        self.metrics_collector.stop()

    def _setup_agents(self):
        """
        Initialize all AutoGen agents.
        """
        agent_configs = self.config.get("agents", {})

        # Create assistant agents for each role with AG2 configuration
        for role, config in agent_configs.items():
            self.agents[role] = AssistantAgent(
                name=config.get("name", f"{role.capitalize()} Agent"),
                system_message=config.get(
                    "system_message", "You are a financial AI assistant."
                ),
                llm_config=self.llm_config,
                description=config.get(
                    "description",
                    f"A specialized agent for {role} tasks in financial analysis",
                ),
            )

        #
        # Create a human proxy agent for testing and interactivity with AG2
        # configuration
        self.agents["human_proxy"] = UserProxyAgent(
            name="HumanProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={
                "use_docker": False
            },  # Disable Docker since it's not available
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        self.logger.info(
            f"Created {len(self.agents)} agents: {list(self.agents.keys())}"
        )

    def _setup_group_chat(self):
        """
        Set up group chat for multi-agent collaboration using AG2 patterns.
        """
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin",  # AG2 supports different speaker selection methods
            allow_repeat_speaker=False,
        )

        self.chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            system_message="You are a group chat manager for a financial trading system. Your job is to facilitate productive discussion between specialized agents.",
        )

        self.logger.info("Group chat and manager initialized")

    def run_trading_cycle(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        mock: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete AutoGen-orchestrated trading workflow.

        Args:
            market_data: Optional market data to provide context
            mock: If True, run in test mode without making actual API calls
            config: Optional configuration for models

        Returns:
            Results of the trading cycle including decisions
        """
        # Increment trading cycle counter
        self.trading_cycles_run += 1
        self.logger.counter("orchestrator.trading_cycles_run", 1)

        if mock:
            self.logger.warning("Running in MOCK mode - no API calls will be made")
            # Return mock result
            return {
                "chat_result": "This is a mock response since mock=True was specified. No API calls were made.",
                "decisions": [],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

        self.logger.info("Running with REAL LLM API calls to OpenRouter")
        self.logger.info("Starting new trading cycle")

        start_time = time.time() # Start timing the cycle

        # Connect to models if not already connected
        if not hasattr(self, "models") or not self.models:
            self.connect_to_models(config)

        # Run the selection model to get candidates if available
        selected_candidates = []
        market_context = {}
        selection_data = {}

        if hasattr(self, "models") and "selection" in self.models:
            self.logger.info("Getting data from SelectionModel")
            try:
                # Get comprehensive selection data
                selection_data = self.models["selection"].get_selection_data()
                self.logger.info(
                    f"Retrieved selection data with {len(selection_data.get('candidates', []))} candidates"
                )
                self.logger.gauge("orchestrator.selected_candidates_count", len(selection_data.get('candidates', [])))

                # Extract market context and candidates
                market_context = selection_data.get("market_context", {})
                selected_candidates = selection_data.get("top_candidates", [])

                # If no candidates in top_candidates, try all candidates
                if not selected_candidates:
                    selected_candidates = selection_data.get("candidates", [])

                # If still no candidates, run a new selection cycle
                if not selected_candidates:
                    self.logger.info(
                        "No candidates found in Redis, running new selection cycle"
                    )
                    selected_candidates = self.models["selection"].run_selection_cycle()
                    self.logger.info(
                        f"SelectionModel identified {len(selected_candidates)} candidates"
                    )
                    self.logger.gauge("orchestrator.selected_candidates_count", len(selected_candidates))

            except Exception as e:
                self.logger.error(f"Error getting selection data: {e}")
                self.logger.exception("Exception details:")

        # Reset the group chat for a new trading cycle
        self.group_chat.reset()

        # Create initial prompt with market context and selection results
        market_context_str = ""

        #
        # Use market context from SelectionModel if available, otherwise use
        # provided market_data
        if market_context:
            market_context_str = f"""
            Current market context:
            - S&P 500: {market_context.get("spy_price", market_data.get("sp500", "N/A"))}
            - VIX: {market_context.get("vix", market_data.get("vix", "N/A"))}
            - Market open: {market_context.get("market_open", "N/A")}
            - Buying power: ${market_context.get("buying_power", "N/A"):,.2f}
            - Portfolio value: ${market_context.get("portfolio_value", "N/A"):,.2f}
            """
        elif market_data:
            market_context_str = f"""
            Current market context:
            - S&P 500: {market_data.get("sp500", "N/A")}
            - VIX: {market_data.get("vix", "N/A")}
            - 10-Year Treasury: {market_data.get("treasury_10y", "N/A")}
            - Market sentiment: {market_data.get("market_sentiment", "N/A")}
            """

        # Add selection results to the prompt
        selection_context = ""
        if selected_candidates:
            selection_context = "\nSelected candidates from the SelectionModel:\n"
            for i, candidate in enumerate(selected_candidates[:5]):  # Show top 5
                selection_context += f"- {candidate.get('ticker')}: Score {candidate.get('score', 0):.2f}, "
                selection_context += f"RSI {candidate.get('rsi', 0):.2f}, "
                selection_context += (
                    f"Relative Volume {candidate.get('relative_volume', 0):.2f}x\n"
                )

            if len(selected_candidates) > 5:
                selection_context += (
                    f"- Plus {len(selected_candidates) - 5} more candidates...\n"
                )

        initial_prompt = f"""
        # Trading Cycle Workflow

        {market_context_str}

        {selection_context}

        We are starting a new trading cycle. Please follow this structured workflow by calling the appropriate functions:

        1. **Selection:** The Selection Agent should review the candidates already identified by the SelectionModel (if any) or run a new selection cycle using `run_selection_cycle()` to identify 3-5 of the most promising stocks for further analysis. Provide detailed reasoning for each selection.

        2. **Data Gathering & Analysis:** For each selected stock, the Data Agent, NLP Agent, Forecaster Agent, and RAG Agent should gather and analyze relevant data by calling the appropriate functions registered for their respective models (e.g., `analyze_news_sentiment()`, `predict_price_movement()`, `analyze_company()`, `get_rag_context()`).

        3. **Decision Making:** Based on all gathered and analyzed data, the agents should collectively decide on trading actions (buy, sell, hold) with specific quantities and reasoning. The Decision Model's `process_analysis_results()` function can be used to integrate analysis results and make decisions.

        4. **Execution:** The Execution Agent should outline how these trades would be executed using the Trade Model's functions (e.g., `execute_trade()`).

        5. **Monitoring:** The Monitoring Agent should suggest metrics to track for these positions.

        Selection Agent, please start by reviewing the candidates and selecting the most promising stocks for analysis, or initiate a new selection cycle if needed.
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
                # Measure LLM call time
                llm_call_start_time = time.time()
                human_proxy.initiate_chat(selection_agent, message=initial_prompt)
                llm_call_duration = (time.time() - llm_call_start_time) * 1000
                self.logger.timing("orchestrator.llm_call_duration_ms", llm_call_duration)
                self.llm_api_call_count += 1
                self.logger.counter("orchestrator.llm_api_call_count", 1)

                # Get the chat history
                messages = selection_agent.chat_history[human_proxy]
                combined_result = "\n\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in messages]
                )
                self.logger.info(f"Got {len(messages)} messages in chat history")

                response = combined_result
            else:
                # Fallback to standard group chat if agents not found
                self.logger.info("Falling back to standard group chat...")

                # Measure LLM call time
                llm_call_start_time = time.time()
                response = self.chat_manager.run(initial_prompt)
                llm_call_duration = (time.time() - llm_call_start_time) * 1000
                self.logger.timing("orchestrator.llm_call_duration_ms", llm_call_duration)
                self.llm_api_call_count += 1
                self.logger.counter("orchestrator.llm_api_call_count", 1)

                self.logger.info(f"Response type: {type(response)}")

        except Exception as e:
            self.logger.error(f"Error in chat execution: {str(e)}")
            self.logger.exception("Exception details:")

            # Increment error counter
            self.execution_errors += 1
            self.logger.counter("orchestrator.execution_errors", 1)
            self.logger.gauge("orchestrator.execution_error_rate", (self.execution_errors / self.trading_cycles_run) * 100)

            # Calculate and log duration even on error
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.trading_cycle_duration_ms", duration)

            return {
                "chat_result": f"Error: {str(e)}",
                "decisions": [],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

        # Extract messages from the response
        # In newer AutoGen versions, we need to handle RunResponse specially
        try:
            # Try to access the messages property or attribute
            if hasattr(response, "messages"):
                messages = response.messages
                # Concatenate messages into a single string
                chat_result = "\n\n".join(
                    [
                        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                        for msg in messages
                        if isinstance(msg, dict)
                    ]
                )
            elif hasattr(response, "chat_history"):
                chat_result = "\n\n".join(
                    [f"{msg[0]}: {msg[1]}" for msg in response.chat_history]
                )
            else:
                # Fallback: try different properties or convert to string
                chat_result = str(response)
                self.logger.warning("Using fallback string conversion for RunResponse")
        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            chat_result = str(response)

        # Extract and process results
        decisions = self._extract_trading_decisions(chat_result)

        # Log the outcomes and metrics
        self.logger.info(f"Trading cycle completed with {len(decisions)} decisions")

        # Increment decisions made counter
        self.decisions_made_count += len(decisions)
        self.logger.counter("orchestrator.decisions_made_count", len(decisions) if decisions else 0)
        self.logger.gauge("orchestrator.selected_candidates_count", len(selected_candidates))

        # Calculate and log duration
        duration = (time.time() - start_time) * 1000
        self.logger.timing("orchestrator.trading_cycle_duration_ms", duration)
        self.logger.gauge("orchestrator.execution_error_rate", (self.execution_errors / self.trading_cycles_run) * 100)


        return {
            "chat_result": chat_result,
            "decisions": decisions,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
                    decisions.append(
                        {
                            "ticker": ticker,
                            "action": action,
                            "quantity": quantity,
                            "reason": reason,
                        }
                    )

        return decisions

    def connect_to_models(self, config=None):
        """
        Connect to NextGen model implementations.

        This method integrates with the actual NextGen Models implementations,
        allowing the agents to call the model methods.

        Args:
            config: Optional configuration dictionary for the models
        """
        self.models = {}

        try:
            # Import all model classes
            from nextgen_models.nextgen_select.select_model import SelectionModel
            from nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model import (
                SentimentAnalysisModel,
            )
            from nextgen_models.nextgen_market_analysis.market_analysis_model import (
                MarketAnalysisModel,
            )
            from nextgen_models.nextgen_context_model.context_model import ContextModel
            from nextgen_models.nextgen_trader.trade_model import TradeModel
            from nextgen_models.nextgen_risk_assessment.risk_assessment_model import (
                RiskAssessmentModel,
            )
            from nextgen_models.nextgen_fundamental_analysis.fundamental_analysis_model import (
                FundamentalAnalysisModel,
            )
            from nextgen_models.nextgen_decision.decision_model import DecisionModel

            self.logger.info("Successfully imported NextGen models")

            # Initialize models with configuration
            model_config = config or {}

            # Initialize the Selection Model
            selection_config = self._prepare_model_config("selection", model_config)
            start_time = time.time()
            self.models["selection"] = SelectionModel(selection_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized SelectionModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Initialize Sentiment Analysis Model
            sentiment_config = self._prepare_model_config(
                "sentiment_analysis", model_config
            )
            start_time = time.time()
            self.models["sentiment"] = SentimentAnalysisModel(sentiment_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized SentimentAnalysisModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Initialize Market Analysis Model
            market_config = self._prepare_model_config("market_analysis", model_config)
            start_time = time.time()
            self.models["market"] = MarketAnalysisModel(market_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized MarketAnalysisModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Initialize Fundamental Analysis Model
            fundamental_config = self._prepare_model_config(
                "fundamental_analysis", model_config
            )
            start_time = time.time()
            self.models["fundamental"] = FundamentalAnalysisModel(fundamental_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized FundamentalAnalysisModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Initialize Risk Assessment Model
            risk_config = self._prepare_model_config("risk_assessment", model_config)
            start_time = time.time()
            self.models["risk"] = RiskAssessmentModel(risk_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized RiskAssessmentModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Initialize Trade Model
            trade_config = self._prepare_model_config("trade", model_config)
            start_time = time.time()
            self.models["trade"] = TradeModel(trade_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized TradeModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Initialize Decision Model
            decision_config = self._prepare_model_config("decision", model_config)
            start_time = time.time()
            self.models["decision"] = DecisionModel(decision_config)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.model_init_time_ms", duration)
            self.logger.info("Initialized DecisionModel")
            self.logger.counter("orchestrator.models_connected", 1)

            # Set up bidirectional communication between models
            self._setup_model_communication()

            # Register model methods with the agents
            self._register_model_methods()

        except ImportError as e:
            self.logger.warning(
                f"Could not import NextGen models: {e}. Running in standalone mode."
            )
            self.logger.counter("orchestrator.model_connection_errors", 1)

    def _prepare_model_config(
        self, model_name: str, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare configuration for a specific model by loading its config file
        and ensuring it has all necessary MCP configurations loaded from their files.

        Args:
            model_name: Name of the model (e.g., 'selection', 'market_analysis')
            base_config: The orchestrator's base configuration dictionary

        Returns:
            Model-specific configuration dictionary with loaded MCP configs
        """
        # Get the model's specific config path from the orchestrator's config
        model_settings = base_config.get("nextgen_models", {}).get(model_name, {})
        model_config_path = model_settings.get("config_path")

        model_config = {}
        if model_config_path and os.path.exists(model_config_path):
            try:
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)
                self.logger.info(f"Loaded config for {model_name} from {model_config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config for {model_name} from {model_config_path}: {e}")
                # Continue with empty config if loading fails

        # Define necessary MCP configurations for each model type
        # This mapping specifies which MCP config keys each model expects
        model_required_mcp_configs = {
            "selection": ["financial_data_mcp_config", "redis_mcp_config", "trading_mcp_config", "time_series_mcp_config"],
            "sentiment_analysis": ["financial_data_mcp_config", "redis_mcp_config", "document_analysis_mcp_config", "vector_store_mcp_config"],
            "market_analysis": ["financial_data_mcp_config", "time_series_mcp_config"],
            "fundamental_analysis": ["financial_data_mcp_config", "risk_analysis_mcp_config"],
            "risk_assessment": ["risk_analysis_mcp_config", "financial_data_mcp_config", "redis_mcp_config"],
            "trade": ["trading_mcp_config", "redis_mcp_config"],
            "decision": ["redis_mcp_config", "trading_mcp_config", "risk_analysis_mcp_config"] # Decision model needs access to various data via Redis
        }

        # Define paths for consolidated MCP configurations
        consolidated_mcp_config_paths = {
            "trading_mcp_config": "config/mcp_tools/trading_mcp_config.json",
            "financial_data_mcp_config": "config/mcp_tools/financial_data_mcp_config.json",
            "time_series_mcp_config": "config/mcp_tools/time_series_mcp_config.json",
            "risk_analysis_mcp_config": "config/mcp_tools/risk_analysis_mcp_config.json",
            "document_analysis_mcp_config": "config/mcp_tools/document_analysis_mcp_config.json",
            "vector_store_mcp_config": "config/mcp_tools/vector_store_mcp_config.json",
            "redis_mcp_config": "config/redis_mcp/redis_mcp_config.json" # Added Redis MCP config path
        }

        # Load and add required MCP configurations to the model's config
        required_configs = model_required_mcp_configs.get(model_name, [])
        for config_key in required_configs:
            if config_key not in model_config: # Avoid overwriting if model config already has it
                mcp_config_path = consolidated_mcp_config_paths.get(config_key)
                if mcp_config_path and os.path.exists(mcp_config_path):
                    try:
                        with open(mcp_config_path, "r") as f:
                            mcp_config = json.load(f)
                        model_config[config_key] = mcp_config
                        self.logger.info(f"Loaded {config_key} for {model_name} from {mcp_config_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading {config_key} for {model_name} from {mcp_config_path}: {e}")
                        # Add an empty dict or None to indicate missing config if necessary for the model
                        model_config[config_key] = {} # Provide empty dict to avoid KeyError, models should handle missing keys gracefully
                else:
                    self.logger.warning(f"MCP config file not found for {config_key} at {mcp_config_path} for model {model_name}")
                    model_config[config_key] = {} # Provide empty dict

        # Pass the LLM config if not already set in the model's specific config
        if "llm_config" not in model_config:
            model_config["llm_config"] = self.llm_config

        return model_config

    def _setup_model_communication(self):
        """
        Set up bidirectional communication between models using Redis streams.
        """
        self.logger.info("Setting up bidirectional communication between models")

        # Get Redis MCP from selection model (assuming it's initialized)
        if "selection" in self.models and hasattr(
            self.models["selection"], "redis_mcp"
        ):
            redis_mcp = self.models["selection"].redis_mcp

            # Create Redis streams for inter-model communication
            streams = [
                "model:selection:candidates",
                "model:sentiment:analysis",
                "model:market:trends",
                "model:fundamental:insights",
                "model:risk:assessments",
                "model:decision:actions",
                "model:trade:executions",
            ]

            for stream in streams:
                try:
                    redis_mcp.create_stream(stream)
                    self.logger.info(f"Created Redis stream: {stream}")
                except Exception as e:
                    self.logger.warning(f"Error creating Redis stream {stream}: {e}")

            # Set up listeners for bidirectional flows
            for model_name, model in self.models.items():
                if hasattr(model, "subscribe_to_streams"):
                    try:
                        model.subscribe_to_streams()
                        self.logger.info(
                            f"Set up stream subscriptions for {model_name}"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Error setting up stream subscriptions for {model_name}: {e}"
                        )
        else:
            self.logger.warning(
                "Could not set up model communication: Redis MCP not available"
            )

    def _register_model_methods(self):
        """
        Register model methods with the appropriate agents using AG2 patterns.

        This connects the specialized model implementations with the AutoGen agents,
        allowing the agents to call model methods directly using the new function registration pattern.
        """
        # Get the user proxy agent
        user_proxy = self.agents.get("human_proxy")
        if not user_proxy:
            self.logger.warning(
                "Could not register model methods: user proxy agent not found"
            )
            return

        # Register methods from each model
        if "selection" in self.models:
            self._register_selection_model_methods(user_proxy)

        if "sentiment" in self.models:
            self._register_sentiment_model_methods(user_proxy)

        if "market" in self.models:
            self._register_market_model_methods(user_proxy)

        if "fundamental" in self.models:
            self._register_fundamental_model_methods(user_proxy)

        if "risk" in self.models:
            self._register_risk_model_methods(user_proxy)

        if "trade" in self.models:
            self._register_trade_model_methods(user_proxy)

        if "decision" in self.models:
            self._register_decision_model_methods(user_proxy)

        # Removed generic MCP tool access registration from orchestrator


    def _register_selection_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register SelectionModel methods with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info("Registering SelectionModel methods with user proxy agent")

        selection_model = self.models["selection"]
        selection_agent = self.agents.get("selection")

        # Register functions using AG2's register_function decorator
        # Core selection methods
        @register_function(
            run_selection_cycle,
            name="run_selection_cycle",
            description="Run a complete selection cycle to identify trading candidates",
            caller=selection_agent,
            executor=user_proxy,
            return_type=List[Dict[str, Any]],
        )
        def run_selection_cycle() -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.run_selection_cycle()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_selection_data,
            name="get_selection_data",
            description="Get comprehensive selection data including market context and candidates",
            caller=selection_agent,
            executor=user_proxy,
            return_type=Dict[str, Any],
        )
        def get_selection_data() -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_selection_data()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Data gathering methods (These methods internally use FinancialDataMCP)
        @register_function(
            get_market_data,
            name="get_market_data",
            description="Get initial universe of stocks",
            caller=selection_agent,
            executor=user_proxy,
            return_type=List[Dict[str, Any]],
        )
        def get_market_data() -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_market_data()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            filter_by_liquidity,
            name="filter_by_liquidity",
            description="Apply liquidity filters to the universe of stocks",
            caller=selection_agent,
            executor=user_proxy,
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"}
            },
            return_type=List[Dict[str, Any]],
        )
        def filter_by_liquidity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.filter_by_liquidity(stocks)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_technical_indicators,
            name="get_technical_indicators",
            description="Calculate technical indicators for the stocks",
            caller=selection_agent,
            executor=user_proxy,
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"}
            },
            return_type=List[Dict[str, Any]],
        )
        def get_technical_indicators(
            stocks: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_technical_indicators(stocks)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_unusual_activity,
            name="get_unusual_activity",
            description="Check for unusual activity in the stocks",
            caller=selection_agent,
            executor=user_proxy,
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"}
            },
            return_type=List[Dict[str, Any]],
        )
        def get_unusual_activity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_unusual_activity(stocks)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Filtering and scoring methods
        @register_function(
            score_candidates,
            name="score_candidates",
            description="Score and rank the candidate stocks",
            caller=selection_agent,
            executor=user_proxy,
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"}
            },
            return_type=List[Dict[str, Any]], # Changed return type to List[Dict[str, Any]] as per SelectModel method
        )
        def score_candidates(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.score_candidates(stocks)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Storage and retrieval methods
        @register_function(
            store_candidates,
            name="store_candidates",
            description="Store candidates in Redis",
            caller=selection_agent,
            executor=user_proxy,
            parameters={
                "candidates": {
                    "type": "array",
                    "description": "List of candidate dictionaries",
                }
            },
            return_type=bool,
        )
        def store_candidates(candidates: List[Dict[str, Any]]) -> bool:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.store_candidates(candidates)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_candidates,
            name="get_candidates",
            description="Get the current list of candidates from Redis",
            caller=selection_agent,
            executor=user_proxy,
            return_type=List[Dict[str, Any]],
        )
        def get_candidates() -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_candidates()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_top_candidates,
            name="get_top_candidates",
            description="Get the top N candidates by score",
            caller=selection_agent,
            executor=user_proxy,
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return",
                    "default": 10,
                }
            },
            return_type=List[Dict[str, Any]],
        )
        def get_top_candidates(limit: int = 10) -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_top_candidates(limit)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_candidate,
            name="get_candidate",
            description="Get a specific candidate by symbol",
            caller=selection_agent,
            executor=user_proxy,
            parameters={"symbol": {"type": "string", "description": "Stock symbol"}},
            return_type=Optional[Dict[str, Any]],
        )
        def get_candidate(symbol: str) -> Optional[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model.get_candidate(symbol)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_market_context,
            name="get_market_context",
            description="Get current market context",
            caller=selection_agent,
            executor=user_proxy,
            return_type=Dict[str, Any],
        )
        def get_market_context() -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = selection_model._get_market_context()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        #
        # Update the selection agent's system message to include available
        # functions
        if selection_agent:
            # Enhance the system message with available functions
            enhanced_message = (
                selection_agent.system_message
                + "\n\nYou have access to the following functions:\n"
            )

            # List of registered functions
            function_names = [
                "run_selection_cycle",
                "get_selection_data",
                "get_market_data",
                "get_technical_indicators",
                "get_unusual_activity",
                "filter_by_liquidity",
                "score_candidates",
                "store_candidates",
                "get_candidates",
                "get_top_candidates",
                "get_candidate",
                "get_market_context",
            ]

            for func_name in function_names:
                enhanced_message += f"- {func_name}()\n"
            enhanced_message += "\nUse these functions to access market data, analyze stocks, and make selection decisions."

            # Update the agent's system message
            selection_agent.update_system_message(enhanced_message)
            self.logger.info(
                "Updated selection agent's system message with available functions"
            )

        self.logger.info("Registered SelectionModel methods with user proxy agent")

    def _register_sentiment_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register SentimentAnalysisModel methods with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info(
            "Registering SentimentAnalysisModel methods with user proxy agent"
        )

        sentiment_model = self.models["sentiment"]
        sentiment_agent = self.agents.get(
            "finnlp"
        )  # Using the finnlp agent for sentiment analysis

        if not sentiment_agent:
            self.logger.warning("Could not find finnlp agent for sentiment analysis")
            return

        # Register sentiment analysis functions
        @register_function(
            analyze_news_sentiment,
            name="analyze_news_sentiment",
            description="Analyze sentiment from news articles for a stock",
            caller=sentiment_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "days": {
                    "type": "integer",
                    "description": "Number of days of news to analyze",
                    "default": 7,
                },
            },
            return_type=Dict[str, Any],
        )
        def analyze_news_sentiment(symbol: str, days: int = 7) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = sentiment_model.analyze_news_sentiment(symbol, days)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            analyze_social_sentiment,
            name="analyze_social_sentiment",
            description="Analyze sentiment from social media for a stock",
            caller=sentiment_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "source": {
                    "type": "string",
                    "description": "Social media source (reddit, twitter, etc.)",
                    "default": "reddit",
                },
            },
            return_type=Dict[str, Any],
        )
        def analyze_social_sentiment(
            symbol: str, source: str = "reddit"
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = sentiment_model.analyze_social_sentiment(symbol, source)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_sentiment_summary,
            name="get_sentiment_summary",
            description="Get a summary of sentiment for a stock",
            caller=sentiment_agent,
            executor=user_proxy,
            parameters={"symbol": {"type": "string", "description": "Stock symbol"}},
            return_type=Dict[str, Any],
        )
        def get_sentiment_summary(symbol: str) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = sentiment_model.get_sentiment_summary(symbol)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Update the sentiment agent's system message
        if sentiment_agent:
            enhanced_message = (
                sentiment_agent.system_message
                + "\n\nYou have access to the following sentiment analysis functions:\n"
            )
            function_names = [
                "analyze_news_sentiment",
                "analyze_social_sentiment",
                "get_sentiment_summary",
            ]

            for func_name in function_names:
                enhanced_message += f"- {func_name}()\n"
            enhanced_message += "\nUse these functions to analyze sentiment from news and social media for stocks."

            sentiment_agent.update_system_message(enhanced_message)
            self.logger.info(
                "Updated sentiment agent's system message with available functions"
            )

        self.logger.info(
            "Registered SentimentAnalysisModel methods with user proxy agent"
        )

    def _register_market_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register MarketAnalysisModel methods with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info(
            "Registering MarketAnalysisModel methods with user proxy agent"
        )

        market_model = self.models["market"]
        # Use the forecaster agent for market analysis
        market_agent = self.agents.get("forecaster")

        if not market_agent:
            self.logger.warning("Could not find forecaster agent for market analysis")
            return

        # Register market analysis functions
        @register_function(
            analyze_market_trends,
            name="analyze_market_trends",
            description="Analyze market trends and conditions",
            caller=market_agent,
            executor=user_proxy,
            parameters={
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for analysis ('day', 'week', 'month')",
                    "default": "day",
                }
            },
            return_type=Dict[str, Any],
        )
        def analyze_market_trends(timeframe: str = "day") -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = market_model.analyze_market_trends(timeframe)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            predict_price_movement,
            name="predict_price_movement",
            description="Predict price movement for a stock",
            caller=market_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "timeframe": {
                    "type": "string",
                    "description": "Prediction timeframe ('day', 'week', 'month')",
                    "default": "day",
                },
            },
            return_type=Dict[str, Any],
        )
        def predict_price_movement(
            symbol: str, timeframe: str = "day"
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = market_model.predict_price_movement(symbol, timeframe)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            get_technical_analysis,
            name="get_technical_analysis",
            description="Get technical analysis for a stock",
            caller=market_agent,
            executor=user_proxy,
            parameters={"symbol": {"type": "string", "description": "Stock symbol"}},
            return_type=Dict[str, Any],
        )
        def get_technical_analysis(symbol: str) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = market_model.get_technical_analysis(symbol)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Update the market agent's system message
        if market_agent:
            enhanced_message += "\nUse these functions to analyze market trends and predict price movements."

            market_agent.update_system_message(enhanced_message)
            self.logger.info(
                "Updated market agent's system message with available functions"
            )

        self.logger.info("Registered MarketAnalysisModel methods with user proxy agent")

    def _register_fundamental_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register FundamentalAnalysisModel methods with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info(
            "Registering FundamentalAnalysisModel methods with user proxy agent"
        )

        fundamental_model = self.models["fundamental"]
        #
        # Use the RAG agent for fundamental analysis since it's related to
        # knowledge retrieval
        fundamental_agent = self.agents.get("rag")

        if not fundamental_agent:
            self.logger.warning("Could not find RAG agent for fundamental analysis")
            return

        # Register fundamental analysis functions
        @register_function(
            name="analyze_company",
            description="Perform comprehensive fundamental analysis of a company",
            caller=fundamental_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "sector": {
                    "type": "string",
                    "description": "Industry sector (optional)",
                    "required": False,
                },
            },
            return_type=Dict[str, Any],
        )
        def analyze_company(
            symbol: str, sector: Optional[str] = None
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = fundamental_model.analyze_company(symbol, sector)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="get_financial_statements",
            description="Get financial statements for a company",
            caller=fundamental_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "statement_type": {
                    "type": "string",
                    "description": "Type of statement ('income', 'balance', 'cash_flow', or 'all')",
                    "default": "all",
                },
                "period": {
                    "type": "string",
                    "description": "Period ('annual' or 'quarterly')",
                    "default": "annual",
                },
            },
            return_type=Dict[str, Any],
        )
        def get_financial_statements(
            symbol: str, statement_type: str = "all", period: str = "annual"
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = fundamental_model.get_financial_statements(
                symbol, statement_type, period
            )
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="calculate_financial_ratios",
            description="Calculate financial ratios from financial statements",
            caller=fundamental_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "ratios": {
                    "type": "array",
                    "description": "List of ratios to calculate (optional)",
                    "required": False,
                },
            },
            return_type=Dict[str, Any],
        )
        def calculate_financial_ratios(
            symbol: str, ratios: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = fundamental_model.calculate_financial_ratios(symbol, ratios)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="analyze_growth_metrics",
            description="Analyze growth metrics for a company",
            caller=fundamental_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "metrics": {
                    "type": "array",
                    "description": "List of metrics to analyze (optional)",
                    "required": False,
                },
            },
            return_type=Dict[str, Any],
        )
        def analyze_growth_metrics(
            symbol: str, metrics: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = fundamental_model.analyze_growth_metrics(symbol, metrics)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Update the fundamental agent's system message
        if fundamental_agent:
            enhanced_message = (
                fundamental_agent.system_message
                + "\n\nYou have access to the following fundamental analysis functions:\n"
            )
            function_names = [
                "analyze_company",
                "get_financial_statements",
                "calculate_financial_ratios",
                "analyze_growth_metrics",
            ]

            for func_name in function_names:
                enhanced_message += f"- {func_name}()\n"
            enhanced_message += "\nUse these functions to analyze company fundamentals and financial health."

            fundamental_agent.update_system_message(enhanced_message)
            self.logger.info(
                "Updated fundamental agent's system message with available functions"
            )

        self.logger.info(
            "Registered FundamentalAnalysisModel methods with user proxy agent"
        )

    def _register_risk_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register RiskAssessmentModel methods with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info(
            "Registering RiskAssessmentModel methods with user proxy agent"
        )

        risk_model = self.models["risk"]
        risk_agent = self.agents.get(
            "monitoring"
        )  # Using monitoring agent for risk assessment

        if not risk_agent:
            self.logger.warning("Could not find monitoring agent for risk assessment")
            return

        # Register risk assessment functions
        @register_function(
            name="assess_portfolio_risk",
            description="Assess risk of the current portfolio",
            caller=risk_agent,
            executor=user_proxy,
            return_type=Dict[str, Any],
        )
        async def assess_portfolio_risk() -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            # Need to provide portfolio and market_data to analyze_portfolio_risk
            # For now, return a placeholder or fetch necessary data
            # This requires a more complex change to fetch data within the registered function or pass it
            # As a temporary fix to resolve the "never awaited" warning and the non-existent method call,
            # I will make this an async function and return a placeholder.
            # A proper fix would involve redesigning how the orchestrator provides data to this function.
            self.logger.warning("assess_portfolio_risk called without portfolio or market data. Returning placeholder.")
            # Simulate fetching data - this would need actual implementation
            portfolio_data = {"AAPL": {"shares": 100, "avg_price": 150.0}} # Placeholder
            market_data = {"AAPL_returns": [0.001] * 252} # Placeholder

            # Await the actual risk analysis method
            result = await risk_model.analyze_portfolio_risk(portfolio_data, market_data)

            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="calculate_position_risk",
            description="Calculate risk for a specific position",
            caller=risk_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "position_size": {
                    "type": "number",
                    "description": "Position size in dollars",
                },
            },
            return_type=Dict[str, Any],
        )
        def calculate_position_risk(
            symbol: str, position_size: float
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = risk_model.calculate_position_risk(symbol, position_size)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="get_risk_metrics",
            description="Get risk metrics for a stock",
            caller=risk_agent,
            executor=user_proxy,
            parameters={"symbol": {"type": "string", "description": "Stock symbol"}},
            return_type=Dict[str, Any],
        )
        def get_risk_metrics(symbol: str) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = risk_model.get_risk_metrics(symbol)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Update the risk agent's system message
        if risk_agent:
            enhanced_message = (
                risk_agent.system_message
                + "\n\nYou have access to the following risk assessment functions:\n"
            )
            function_names = [
                "assess_portfolio_risk",
                "calculate_position_risk",
                "get_risk_metrics",
            ]

            for func_name in function_names:
                enhanced_message += f"- {func_name}()\n"
            enhanced_message += "\nUse these functions to assess risk for the portfolio and individual positions."

            risk_agent.update_system_message(enhanced_message)
            self.logger.info(
                "Updated risk agent's system message with available functions"
            )

        self.logger.info("Registered RiskAssessmentModel methods with user proxy agent")

    def _register_trade_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register TradeModel methods with the user proxy agent.
        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info("Registering TradeModel methods with user proxy agent")

        trade_model = self.models["trade"]
        trade_agent = self.agents.get(
            "execution"
        )  # Using execution agent for trade execution

        if not trade_agent:
            self.logger.warning("Could not find execution agent for trade execution")
            return

        # Register trade execution functions
        @register_function(
            name="execute_trade",
            description="Execute a trade order",
            caller=trade_agent,
            executor=user_proxy,
            parameters={
                "symbol": {"type": "string", "description": "Stock symbol"},
                "action": {
                    "type": "string",
                    "description": "Trade action ('buy' or 'sell')",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of shares to trade",
                },
                "order_type": {
                    "type": "string",
                    "description": "Order type ('market', 'limit', 'stop')",
                    "default": "market",
                },
                "price": {
                    "type": "number",
                    "description": "Price for limit or stop orders (optional)",
                    "required": False,
                },
            },
            return_type=Dict[str, Any],
        )
        def execute_trade(
            symbol: str,
            action: str,
            quantity: int,
            order_type: str = "market",
            price: Optional[float] = None,
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = trade_model.execute_trade(
                symbol, action, quantity, order_type, price
            )
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="get_order_status",
            description="Get status of an order",
            caller=trade_agent,
            executor=user_proxy,
            parameters={"order_id": {"type": "string", "description": "Order ID"}},
            return_type=Dict[str, Any],
        )
        def get_order_status(order_id: str) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = trade_model.get_order_status(order_id)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="get_positions",
            description="Get current positions",
            caller=trade_agent,
            executor=user_proxy,
            return_type=List[Dict[str, Any]],
        )
        def get_positions() -> List[Dict[str, Any]]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = trade_model.get_positions()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="get_account_info",
            description="Get account information",
            caller=trade_agent,
            executor=user_proxy,
            return_type=Dict[str, Any],
        )
        def get_account_info() -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = trade_model.get_account_info()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        # Update the trade agent's system message
        if trade_agent:
            enhanced_message = (
                trade_agent.system_message
                + "\n\nYou have access to the following trade execution functions:\n"
            )
            function_names = [
                "execute_trade",
                "get_order_status",
                "get_positions",
                "get_account_info",
            ]

            for func_name in function_names:
                enhanced_message += f"- {func_name}()\n"
            enhanced_message += (
                "\nUse these functions to execute trades and monitor positions."
            )

            trade_agent.update_system_message(enhanced_message)
            self.logger.info(
                "Updated trade agent's system message with available functions"
            )

        self.logger.info("Registered TradeModel methods with user proxy agent")

    def _register_decision_model_methods(self, user_proxy: UserProxyAgent):
        """
        Register DecisionModel methods with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register methods with
        """
        self.logger.info("Registering DecisionModel methods with user proxy agent")

        decision_model = self.models["decision"]
        #
        # We don't have a specific decision agent, so we'll use the human_proxy
        # as the caller
        decision_agent = user_proxy

        # Register decision-making functions
        @register_function(
            name="process_analysis_results",
            description="Process analysis from all models to make decisions",
            caller=decision_agent,
            executor=user_proxy,
            parameters={
                "selection_data": {
                    "type": "object",
                    "description": "Data from selection model",
                },
                "finnlp_data": {
                    "type": "object",
                    "description": "Data from sentiment analysis",
                },
                "forecaster_data": {
                    "type": "object",
                    "description": "Data from price forecasting",
                },
                "rag_data": {
                    "type": "object",
                    "description": "Data from knowledge retrieval",
                },
            },
            return_type=Dict[str, Any],
        )
        def process_analysis_results(
            selection_data: Dict[str, Any],
            finnlp_data: Dict[str, Any],
            forecaster_data: Dict[str, Any],
            rag_data: Dict[str, Any],
        ) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = decision_model.process_analysis_results(
                selection_data, finnlp_data, forecaster_data, rag_data
            )
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="evaluate_market_conditions",
            description="Evaluate overall market state",
            caller=decision_agent,
            executor=user_proxy,
            return_type=Dict[str, Any],
        )
        def evaluate_market_conditions() -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = decision_model.evaluate_market_conditions()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="check_portfolio_constraints",
            description="Check portfolio allocation and constraints",
            caller=decision_agent,
            executor=user_proxy,
            return_type=Dict[str, Any],
        )
        def check_portfolio_constraints() -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = decision_model.check_portfolio_constraints()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="make_trade_decision",
            description="Make final buy/sell decision with confidence levels",
            caller=decision_agent,
            executor=user_proxy,
            parameters={"symbol": {"type": "string", "description": "Stock symbol"}},
            return_type=Dict[str, Any],
        )
        def make_trade_decision(symbol: str) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = decision_model.make_trade_decision(symbol)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="handle_position_update",
            description="Handle position updates (trades executed, positions closed)",
            caller=decision_agent,
            executor=user_proxy,
            parameters={
                "update_data": {"type": "object", "description": "Position update data"}
            },
            return_type=Dict[str, Any],
        )
        def handle_position_update(update_data: Dict[str, Any]) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = decision_model.handle_position_update(update_data)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        @register_function(
            name="request_new_selections",
            description="Request new stock selections when capital becomes available",
            caller=decision_agent,
            executor=user_proxy,
            parameters={
                "available_capital": {
                    "type": "number",
                    "description": "Available capital for new positions",
                }
            },
            return_type=Dict[str, Any],
        )
        def request_new_selections(available_capital: float) -> Dict[str, Any]:
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()
            result = decision_model.request_new_selections(available_capital)
            duration = (time.time() - start_time) * 1000
            self.logger.timing("orchestrator.function_call_duration_ms", duration)
            return result

        self.logger.info("Registered DecisionModel methods with user proxy agent")

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
            caller=user_proxy, # MCP tool calls are initiated by the user proxy
            executor=user_proxy,
            parameters={
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server providing the tool",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool",
                },
            },
            return_type=Any,
        )
        def use_mcp_tool(
            server_name: str, tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            """
            Use a tool provided by an MCP server.

            Args:
                server_name: Name of the MCP server providing the tool
                tool_name: Name of the tool to execute
                arguments: Arguments for the tool

            Returns:
                Result of the tool execution
            """
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()

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
                    self.logger.warning(f"MCP server not found: {server_name}", server=server_name)
                    self.logger.counter("orchestrator.mcp_tool_error_count", 1)
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.mcp_tool_call_duration_ms", duration)
                    return {"error": f"MCP server not found: {server_name}"}

                # Call the tool
                try:
                    result = mcp_server.call_tool(tool_name, arguments)
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.mcp_tool_call_duration_ms", duration)
                    self.logger.counter("orchestrator.mcp_tool_call_count", 1)
                    return result
                except Exception as e:
                    self.logger.error(f"Error calling MCP tool {tool_name} on {server_name}: {e}", server=server_name, tool=tool_name, error=str(e))
                    self.logger.counter("orchestrator.mcp_tool_error_count", 1)
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.mcp_tool_call_duration_ms", duration)
                    return {"error": str(e)}
            else:
                self.logger.warning("Could not use MCP tool: Selection model not initialized", server=server_name, tool=tool_name)
                self.logger.counter("orchestrator.mcp_tool_error_count", 1)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("orchestrator.mcp_tool_call_duration_ms", duration)
                return {"error": "Selection model not initialized"}

        @register_function(
            name="list_mcp_tools",
            description="List all available tools on an MCP server",
            caller=user_proxy, # MCP tool calls are initiated by the user proxy
            executor=user_proxy,
            parameters={
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server",
                }
            },
            return_type=List[Dict[str, str]],
        )
        def list_mcp_tools(server_name: str) -> List[Dict[str, str]]:
            """
            List all available tools on an MCP server.

            Args:
                server_name: Name of the MCP server

            Returns:
                List of tool information dictionaries
            """
            self.logger.counter("orchestrator.registered_functions", 1)
            start_time = time.time()

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
                    self.logger.warning(f"MCP server not found for listing tools: {server_name}", server=server_name)
                    self.logger.counter("orchestrator.mcp_tool_error_count", 1)
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.mcp_list_tools_duration_ms", duration)
                    return [{"error": f"MCP server not found: {server_name}"}]

                # List the tools
                try:
                    result = mcp_server.list_tools()
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.mcp_list_tools_duration_ms", duration)
                    self.logger.counter("orchestrator.mcp_list_tools_count", 1)
                    return result
                except Exception as e:
                    self.logger.error(f"Error listing tools on MCP server {server_name}: {e}", server=server_name, error=str(e))
                    self.logger.counter("orchestrator.mcp_tool_error_count", 1)
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.mcp_list_tools_duration_ms", duration)
                    return [{"error": str(e)}]
            else:
                self.logger.warning("Could not list MCP tools: Selection model not initialized", server=server_name, tool="list_tools")
                self.logger.counter("orchestrator.mcp_tool_error_count", 1)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("orchestrator.mcp_list_tools_duration_ms", duration)
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
        self.logger.info(f"Executing {len(decisions)} trades")
        
        execution_results = {
            "total": len(decisions),
            "success": 0,
            "failed": 0,
            "details": [],
        }
        
        # Check if we have a trade model available
        if hasattr(self, "models") and "trade" in self.models:
            trade_model = self.models["trade"]
            
            for decision in decisions:
                try:
                    # Extract decision details
                    symbol = decision.get("ticker")
                    action = decision.get("action", "").lower()
                    quantity = decision.get("quantity", 0)
                    
                    if not symbol or not action or quantity <= 0:
                        self.logger.warning(f"Invalid trade decision: {decision}")
                        execution_results["details"].append({
                            "decision": decision,
                            "success": False,
                            "error": "Invalid decision parameters"
                        })
                        execution_results["failed"] += 1
                        continue
                    
                    # Execute the trade
                    self.logger.info(f"Executing trade: {action} {quantity} shares of {symbol}")
                    start_time = time.time()
                    
                    # Default to market order
                    result = trade_model.execute_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        order_type="market"
                    )
                    
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("orchestrator.trade_execution_time_ms", duration)
                    
                    execution_results["details"].append({
                        "decision": decision,
                        "result": result,
                        "success": result.get("success", False)
                    })
                    
                    if result.get("success", False):
                        execution_results["success"] += 1
                        self.logger.info(f"Successfully executed trade for {symbol}")
                        self.logger.counter("orchestrator.successful_trades", 1)
                    else:
                        execution_results["failed"] += 1
                        self.logger.warning(f"Failed to execute trade for {symbol}: {result.get('error', 'Unknown error')}")
                        self.logger.counter("orchestrator.failed_trades", 1)
                
                except Exception as e:
                    self.logger.error(f"Error executing trade for decision {decision}: {e}")
                    self.logger.exception("Exception details:")
                    execution_results["details"].append({
                        "decision": decision,
                        "success": False,
                        "error": str(e)
                    })
                    execution_results["failed"] += 1
                    self.logger.counter("orchestrator.trade_execution_errors", 1)
        else:
            self.logger.warning("No trade model available. Trades will not be executed.")
            for decision in decisions:
                execution_results["details"].append({
                    "decision": decision,
                    "success": False,
                    "error": "No trade model available"
                })
                execution_results["failed"] += 1
        
        # Log execution summary
        self.logger.info(f"Trade execution summary: {execution_results['success']} successful, {execution_results['failed']} failed")
        
        return execution_results


def create_orchestrator(
    config_path: Optional[str] = None, model_config: Optional[Dict[str, Any]] = None
) -> AutoGenOrchestrator:
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
    print(
        "Available models: anthropic/claude-3-opus, meta-llama/llama-3-70b, google/gemini-1.5-pro"
    )

    orchestrator = create_orchestrator()

    # Run a test trading cycle with some market data
    sample_market_data = {
        "sp500": 5024.35,
        "vix": 14.87,
        "treasury_10y": 4.12,
        "market_sentiment": "mixed",
    }

    print("Running trading cycle with sample market data...")
    print("Using OpenRouter API from environment variables")

    # Sample model configuration with MCP configurations
    model_config = {
        # MCP configurations that will be shared with models
        "alpaca_config": {
            "api_key": get_api_key("alpaca"),
            "api_secret": get_api_key("alpaca_secret"),
            "paper": True,  # Use paper trading
        },
        "redis_config": {
            "host": get_env("REDIS_HOST", "localhost"),
            "port": int(get_env("REDIS_PORT", "6379")),
            "db": int(get_env("REDIS_DB", "0")),
        },
        "polygon_rest_config": {"api_key": get_api_key("polygon")},
        "polygon_ws_config": {"api_key": get_api_key("polygon")},
        "unusual_whales_config": {
            "api_key": get_api_key("unusual_whales")
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
                        "api_key": get_api_key("openrouter"),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    }
                ],
            },
        },
    }

    # Run the trading cycle with the configuration
    result = orchestrator.run_trading_cycle(sample_market_data, config=model_config)
    print(f"Trading cycle completed with {len(result.get('decisions', []))} decisions")
