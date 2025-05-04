"""
AutoGen-Powered Orchestrator for FinGPT

This module implements the central orchestrator for the FinGPT system using Microsoft's AutoGen framework.
The orchestrator coordinates various specialized agents to perform financial analysis and trading decisions.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


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
            "timeout": llm_section.get("timeout", 600)
            # Note: Removed "cache" parameter which was causing validation errors
        }

    def _setup_agents(self):
        """
        Initialize all AutoGen agents.
        """
        agent_configs = self.config.get("agents", {})
        
        # Create assistant agents for each role
        for role, config in agent_configs.items():
            self.agents[role] = AssistantAgent(
                name=config.get("name", f"{role.capitalize()} Agent"),
                system_message=config.get("system_message", "You are a financial AI assistant."),
                llm_config=self.llm_config
            )
            
        # Create a human proxy agent for testing and interactivity
        self.agents["human_proxy"] = UserProxyAgent(
            name="HumanProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"use_docker": False}  # Disable Docker since it's not available
        )
        
        self.logger.info(f"Created {len(self.agents)} agents: {list(self.agents.keys())}")

    def _setup_group_chat(self):
        """
        Set up group chat for multi-agent collaboration.
        """
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=10
        )
        
        self.chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )
        
        self.logger.info("Group chat and manager initialized")

    def run_trading_cycle(self, market_data: Optional[Dict[str, Any]] = None, mock: bool = False) -> Dict[str, Any]:
        """
        Execute a complete AutoGen-orchestrated trading workflow.
        
        Args:
            market_data: Optional market data to provide context
            mock: If True, run in test mode without making actual API calls
            
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
        
        # Reset the group chat for a new trading cycle
        self.group_chat.reset()
        
        # Create initial prompt with market context if provided
        market_context = ""
        if market_data:
            market_context = f"""
            Current market context:
            - S&P 500: {market_data.get('sp500', 'N/A')}
            - VIX: {market_data.get('vix', 'N/A')}
            - 10-Year Treasury: {market_data.get('treasury_10y', 'N/A')}
            - Market sentiment: {market_data.get('market_sentiment', 'N/A')}
            """
            
        initial_prompt = f"""
        {market_context}
        
        We are starting a new trading cycle. Please follow this structured workflow:
        
        1. First, the Selection Agent should identify 3-5 promising stocks to analyze today based on the market context and screening criteria.
        
        2. For each selected stock, the Data Agent should gather relevant financial data, including:
           - Recent price movements
           - Trading volumes
           - Key financial metrics
        
        3. The NLP Agent should analyze news sentiment for each selected stock.
        
        4. The Forecaster Agent should provide price predictions for each stock.
        
        5. The RAG Agent should provide relevant historical context about similar market conditions or company events.
        
        6. Based on all inputs, collectively decide on trading actions (buy, sell, hold) with specific quantities and reasoning.
        
        7. The Execution Agent should outline how these trades would be executed.
        
        8. The Monitoring Agent should suggest metrics to track for these positions.
        
        Selection Agent, please start by identifying promising stocks to analyze.
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

    def connect_to_models(self):
        """
        Connect to FinGPT model implementations.
        
        This method would integrate with the actual FinGPT model implementations,
        allowing the agents to call the model methods.
        """
        # Import FinGPT models
        # This is a placeholder for future implementation
        try:
            from fingpt.nextgen_selection.selection_model import SelectionModel
            from fingpt.fingpt_finnlp.finnlp_model import FinNLPModel
            from fingpt.fingpt_forcaster.forcaster_model import ForcasterModel
            from fingpt.fingpt_rag.rag_model import RAGModel
            from fingpt.nextgen_execution.execution_model import ExecutionModel
            
            self.logger.info("Successfully imported FinGPT models")
            
            # In future implementation, would initialize models and connect to agents
            # self.selection_model = SelectionModel()
            # ...
            
        except ImportError as e:
            self.logger.warning(f"Could not import FinGPT models: {e}. Running in standalone mode.")

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


def create_orchestrator(config_path: Optional[str] = None) -> AutoGenOrchestrator:
    """
    Factory function to create and initialize an AutoGenOrchestrator.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized AutoGenOrchestrator
    """
    orchestrator = AutoGenOrchestrator(config_path)
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
    
    result = orchestrator.run_trading_cycle(sample_market_data)
    print(f"Trading cycle completed with {len(result.get('decisions', []))} decisions")