# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Centralized agent management module for the Agentic AI system

from datetime import datetime
from typing import Any, Dict, List

from src.que_agents.agents.personal_virtual_assistant_agent import (
    PersonalVirtualAssistantAgent,
)
from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.utils.config_manager import ConfigManager

AGENT_INITIALIZATION_CONTEXT = "Agent Initialization"
AGENT_CONFIG_FILENAME = "agent_config.yaml"


class AgentManager:
    """Centralized agent management with better error handling"""

    def __init__(self):
        self.agents = {}
        self.agent_configs = {}
        self.agent_status = {
            "customer_support": False,
            "marketing": False,
            "personal_virtual_assistant": False,
            "financial_trading_bot": False,
        }
        self.fallback_agents = {}
        self.config_manager = ConfigManager()
        self._load_agent_configs()

    def initialize_agents(self):
        """Initialize all agents with comprehensive error handling"""
        system_logger.info("Starting agent initialization...")

        # Initialize Customer Support Agent
        self._initialize_customer_support_agent()

        # Initialize Marketing Agent
        self._initialize_marketing_agent()

        # Initialize Personal Virtual Assistant Agent
        self._initialize_pva_agent()

        # Initialize Financial Trading Bot Agent
        self._initialize_trading_bot_agent()

        # Log overall status
        successful_agents = sum(self.agent_status.values())
        total_agents = len(self.agent_status)

        system_logger.info(
            f"Agent initialization completed: {successful_agents}/{total_agents} agents initialized",
            additional_info={
                "agents_status": self.agent_status,
                "successful_agents": successful_agents,
                "total_agents": total_agents,
            },
        )

    def _initialize_customer_support_agent(self):
        """Initialize Customer Support Agent"""
        try:
            from src.que_agents.agents.customer_support_agent import (
                CustomerSupportAgent,
            )

            # Store as a class reference, not an instance
            self.agents["customer_support"] = CustomerSupportAgent
            self.agent_status["customer_support"] = True
            system_logger.info("Customer Support Agent initialized successfully")
        except Exception as e:
            system_logger.error(
                "Failed to initialize Customer Support Agent",
                additional_info={
                    "context": AGENT_INITIALIZATION_CONTEXT,
                    "agent": "CustomerSupportAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["customer_support"] = None
            self._setup_customer_support_fallback()

    def _initialize_marketing_agent(self):
        """Initialize Marketing Agent with enhanced error handling"""
        try:
            from src.que_agents.agents.marketing_agent import MarketingAgent

            # Test if agent can be instantiated
            test_agent = MarketingAgent()

            # Test basic functionality
            if hasattr(test_agent, "create_marketing_campaign") and callable(
                test_agent.create_marketing_campaign
            ):
                from src.que_agents.agents.marketing_agent import MarketingAgent
                self.agents["marketing"] = MarketingAgent
                self.agent_status["marketing"] = True
                system_logger.info("Marketing Agent initialized successfully")
            else:
                raise AttributeError("Agent missing required methods")

        except Exception as e:
            system_logger.error(
                "Failed to initialize Marketing Agent",
                additional_info={
                    "context": AGENT_INITIALIZATION_CONTEXT,
                    "agent": "MarketingAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["marketing"] = None
            self._setup_marketing_fallback()

    def _initialize_pva_agent(self):
        try:
            config = self.config_manager.load_config(AGENT_CONFIG_FILENAME)
        except FileNotFoundError as e:
            system_logger.error(
                "Failed to load agent configuration file for Personal Virtual Assistant Agent",
                additional_info={
                    "context": AGENT_INITIALIZATION_CONTEXT,
                    "agent": "PersonalVirtualAssistantAgent",
                    "error_type": type(e).__name__,
                    "suggestion": "Create the agent configuration file",
                },
                exc_info=True,
            )
            self.agents["personal_virtual_assistant"] = None
            self._setup_pva_fallback()
            return
        except Exception as e:
            system_logger.error(
                "Failed to initialize Personal Virtual Assistant Agent",
                additional_info={
                    "context": AGENT_INITIALIZATION_CONTEXT,
                    "agent": "PersonalVirtualAssistantAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["personal_virtual_assistant"] = None
            self._setup_pva_fallback()
            return

        # Check if the agent config exists
        agent_config_key = "personal_virtual_assistant"
        if config is None or agent_config_key not in config:
            system_logger.error(
                "Configuration key for Personal Virtual Assistant Agent not found in agent_config.yaml",
                additional_info={
                    "context": AGENT_INITIALIZATION_CONTEXT,
                    "agent": "PersonalVirtualAssistantAgent",
                    "error_type": "KeyError",
                    "suggestion": "Check agent_config.yaml for correct key names",
                },
                exc_info=True,
            )
            self.agents["personal_virtual_assistant"] = None
            self._setup_pva_fallback()
            return

        # Initialize with config
        try:
            test_agent = PersonalVirtualAssistantAgent()

            # Verify agent has required methods
            required_methods = ["handle_user_request", "get_assistant_knowledge"]
            for method in required_methods:
                if not hasattr(test_agent, method):
                    raise AttributeError(f"Agent missing required method: {method}")

            self.agents["personal_virtual_assistant"] = PersonalVirtualAssistantAgent
            self.agent_status["personal_virtual_assistant"] = True
            system_logger.info(
                f"Personal Virtual Assistant Agent initialized successfully with {config[agent_config_key].get('model_name', 'default')} model"
            )
        except Exception as e:
            system_logger.error(
                f"Failed to initialize Personal Virtual Assistant Agent: {str(e)}",
                additional_info={
                    "context": "Agent Initialization",
                    "agent": "PersonalVirtualAssistantAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["personal_virtual_assistant"] = None
            self._setup_pva_fallback()

    def _initialize_trading_bot_agent(self):
        """Initialize Financial Trading Bot Agent with Groq configuration"""
        try:
            config = self.config_manager.load_config("agent_config.yaml")

            agent_config_key = "financial_trading_bot"
            if agent_config_key not in config:
                system_logger.error(
                    "Configuration key for Financial Trading Bot Agent not found in agent_config.yaml",
                    additional_info={
                        "context": AGENT_INITIALIZATION_CONTEXT,
                        "agent": "FinancialTradingBotAgent",
                        "error_type": "KeyError",
                        "suggestion": "Check agent_config.yaml for correct key names",
                    },
                    exc_info=True,
                )
                self.agents["financial_trading_bot"] = None
                self._setup_trading_bot_fallback()
                return

            from src.que_agents.agents.financial_trading_bot_agent import (
                FinancialTradingBotAgent,
            )

            # Initialize with config
            test_agent = FinancialTradingBotAgent()

            # Verify agent has required methods
            required_methods = [
                "make_trading_decision",
                "analyze_market_with_knowledge",
            ]
            for method in required_methods:
                if not hasattr(test_agent, method):
                    system_logger.warning(
                        f"Agent missing method: {method}, but continuing initialization"
                    )

            from src.que_agents.agents.financial_trading_bot_agent import (
                FinancialTradingBotAgent,
            )
            self.agents["financial_trading_bot"] = FinancialTradingBotAgent
            self.agent_status["financial_trading_bot"] = True
            system_logger.info("Financial Trading Bot Agent initialized successfully")
        except Exception as e:
            system_logger.error(
                "Failed to initialize Financial Trading Bot Agent",
                additional_info={
                    "context": AGENT_INITIALIZATION_CONTEXT,
                    "agent": "FinancialTradingBotAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["financial_trading_bot"] = None
            self._setup_trading_bot_fallback()

    def _setup_customer_support_fallback(self):
        """Setup fallback for customer support agent"""

        class FallbackCustomerSupportAgent:
            def handle_customer_request_enhanced(self, _customer_id, _message):
                return {
                    "response": "Customer support is temporarily unavailable. Please try again later.",
                    "confidence": 0.5,
                    "escalate": True,
                    "suggested_actions": [
                        "clear_browser_cache",
                        "contact_support",
                    ],
                    "knowledge_sources": [
                        "login_troubleshooting",
                        "password_reset_guide",
                    ],
                    "sentiment": "neutral",
                }

        self.fallback_agents["customer_support"] = FallbackCustomerSupportAgent()

    def _setup_marketing_fallback(self):
        """Setup fallback for marketing agent"""

        class FallbackMarketingAgent:
            def create_marketing_campaign(self, _request_dict):
                return {
                    "campaign_id": "fallback_001",
                    "status": "created",
                    "message": "Marketing agent is temporarily unavailable. Campaign created with basic template.",
                }

            def generate_marketing_content(self, _request_dict):
                return {
                    "content": "Marketing content generation is temporarily unavailable.",
                    "status": "fallback",
                }

            def analyze_campaign_performance(self, campaign_id):
                return {
                    "campaign_id": campaign_id,
                    "status": "unavailable",
                    "message": "Campaign analysis is temporarily unavailable.",
                }

        self.fallback_agents["marketing"] = FallbackMarketingAgent()

    def _setup_pva_fallback(self):
        """Setup fallback for PVA agent"""

        class FallbackPVAAgent:
            def handle_user_request(
                self, _user_id: str, _user_message: str, _session_id: str = ""
            ):
                return {
                    "response": "Hello! I'm your personal assistant. I'm currently experiencing technical difficulties, but I'm here to help as best I can.",
                    "intent": "greeting",
                    "entities": {},
                    "confidence": 0.5,
                    "actions_taken": ["fallback_response"],
                    "suggestions": ["Try again later", "Contact support"],
                    "timestamp": datetime.now().isoformat(),
                }

            def get_user_context(self, _user_id: str):
                return None

        self.fallback_agents["personal_virtual_assistant"] = FallbackPVAAgent()

    def _setup_trading_bot_fallback(self):
        """Setup fallback for trading bot agent"""

        class FallbackTradingBotAgent:
            def make_trading_decision(
                self, symbol: str, _strategy_type: str = "conservative"
            ):
                from dataclasses import dataclass

                @dataclass
                class TradingDecision:
                    action: str
                    symbol: str
                    quantity: float
                    confidence: float
                    reasoning: str
                    risk_score: float
                    expected_return: float

                return TradingDecision(
                    action="hold",
                    symbol=symbol,
                    quantity=0.0,
                    confidence=0.7,
                    reasoning=f"Trading bot is temporarily unavailable. Recommending hold for {symbol}.",
                    risk_score=0.5,
                    expected_return=0.02,
                )

            def run_trading_cycle(self, symbols: List[str]):
                return {
                    "timestamp": datetime.now().isoformat(),
                    "symbols_analyzed": len(symbols),
                    "trades_executed": 0,
                    "decisions": [
                        {
                            "symbol": symbol,
                            "action": "hold",
                            "confidence": 0.6,
                            "executed": False,
                        }
                        for symbol in symbols
                    ],
                    "portfolio_status": {
                        "total_value": 10000.0,
                        "cash_balance": 5000.0,
                    },
                    "average_confidence": 0.6,
                }

            def get_portfolio_status(self):
                from dataclasses import dataclass

                @dataclass
                class PortfolioStatus:
                    total_value: float
                    cash_balance: float
                    holdings: Dict[str, Any]
                    performance_metrics: Dict[str, Any]
                    unrealized_pnl: float
                    realized_pnl: float

                return PortfolioStatus(
                    total_value=10000.0,
                    cash_balance=5000.0,
                    holdings={},
                    performance_metrics={},
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                )

            def get_performance_report(self):
                return {
                    "portfolio_value": 10000.0,
                    "total_return": 0.0,
                    "trade_statistics": {"total_trades": 0},
                    "recent_trades": [],
                    "status": "fallback_mode",
                }

            def get_market_data(self, symbol: str):
                from dataclasses import dataclass

                @dataclass
                class MarketConditions:
                    symbol: str
                    current_price: float
                    volume: int
                    change_24h: float
                    rsi: float
                    macd: float
                    moving_avg_20: float
                    moving_avg_50: float
                    volatility: float
                    market_sentiment: str

                return MarketConditions(
                    symbol=symbol,
                    current_price=150.0,
                    volume=1000000,
                    change_24h=2.5,
                    rsi=50.0,
                    macd=0.0,
                    moving_avg_20=148.0,
                    moving_avg_50=145.0,
                    volatility=0.2,
                    market_sentiment="neutral",
                )

        self.fallback_agents["financial_trading_bot"] = FallbackTradingBotAgent()

    def get_agent(self, agent_name: str, token: str):
        """Get agent or fallback agent"""
        if not self._is_agent_enabled(agent_name):
            return None

        # If agent class exists, create instance
        if agent_name in self.agents and self.agents[agent_name] is not None:
            try:
                # Create new instance of the agent class
                agent_class = self.agents[agent_name]
                if callable(agent_class):
                    return agent_class()
                else:
                    return agent_class
            except Exception as e:
                system_logger.error(f"Error creating agent instance: {e}")
                return self.fallback_agents.get(agent_name)
        
        return self.fallback_agents.get(agent_name)

    def is_agent_active(self, agent_name: str) -> str:
        """Check agent status"""
        if self.agents.get(agent_name):
            return "active"
        elif self.fallback_agents.get(agent_name):
            return "fallback"
        else:
            return "not_initialized"

    def calculate_agent_health_score(self) -> float:
        """Calculate overall agent health score (0.0 to 1.0)"""
        total_agents = len(self.agent_status)
        if total_agents == 0:
            return 0.0

        active_agents = sum(1 for status in self.agent_status.values() if status)
        fallback_agents = len(self.fallback_agents)

        # Active agents get full score, fallback agents get partial score
        score = (active_agents + (fallback_agents * 0.5)) / total_agents
        return round(min(score, 1.0), 2)

    def _load_agent_configs(self):
        """Load agent configurations"""
        try:
            config = self.config_manager.load_config(AGENT_CONFIG_FILENAME)
            self.agent_configs = config.get("agents", {})
        except Exception:
            self.agent_configs = {}

    def _create_agent(self, agent_type: str):
        """Create agent instance"""
        try:
            if agent_type == "customer_support":
                from src.que_agents.agents.customer_support_agent import (
                    CustomerSupportAgent,
                )

                return CustomerSupportAgent()
            elif agent_type == "marketing":
                from src.que_agents.agents.marketing_agent import MarketingAgent

                return MarketingAgent()
            elif agent_type == "financial_trading_bot":
                from src.que_agents.agents.financial_trading_bot_agent import (
                    FinancialTradingBotAgent,
                )

                return FinancialTradingBotAgent()
            elif agent_type == "personal_virtual_assistant":
                return PersonalVirtualAssistantAgent()
            return None
        except Exception:
            return None

    def remove_agent(self, agent_type: str, token: str) -> bool:
        """Remove agent instance"""
        try:
            if agent_type in self.agents and token in self.agents[agent_type]:
                del self.agents[agent_type][token]
                return True
            return False
        except Exception:
            return False

    def list_agents(self) -> Dict[str, Any]:
        """List all agents"""
        return {
            agent_type: list(instances.keys())
            for agent_type, instances in self.agents.items()
        }

    def get_agent_status(self, agent_type: str, token: str) -> str:
        """Get agent status"""
        if agent_type in self.agents and token in self.agents[agent_type]:
            return "active"
        return "not_found"

    def cleanup_agents(self) -> int:
        """Cleanup agents"""
        cleaned = 0
        try:
            for agent_type in self.agents.keys():
                for token in self.agents[agent_type].keys():
                    try:
                        agent = self.agents[agent_type][token]
                        if hasattr(agent, "cleanup"):
                            agent.cleanup()
                        del self.agents[agent_type][token]
                        cleaned += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return cleaned

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_agents = sum(len(instances) for instances in self.agents.values())
        return {
            "total_agents": total_agents,
            "agent_types": len(self.agents),
            "health_score": self.calculate_agent_health_score(),
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        total_agents = sum(len(instances) for instances in self.agents.values())
        return {
            "status": "healthy" if total_agents > 0 else "no_agents",
            "total_agents": total_agents,
            "timestamp": datetime.now().isoformat(),
        }

    def reload_configs(self) -> bool:
        """Reload configurations"""
        try:
            self._load_agent_configs()
            return True
        except Exception:
            return False

    def _validate_agent_type(self, agent_type: str) -> bool:
        """Validate agent type"""
        valid_types = [
            "customer_support",
            "marketing",
            "financial_trading_bot",
            "personal_virtual_assistant",
        ]
        return agent_type in valid_types

    def _get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.agent_configs.get(agent_type, {})

    def _is_agent_enabled(self, agent_type: str) -> bool:
        """Check if agent is enabled"""
        config = self._get_agent_config(agent_type)
        return config.get("enabled", True)
