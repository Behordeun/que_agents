# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-07 21:56:24
# @Description: This module implements the main API for the Agentic AI system

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

import psutil
import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.que_agents.agents.personal_virtual_assistant_agent import (
    PersonalVirtualAssistantAgent,
)
from src.que_agents.core.schemas import (
    CustomerSupportRequest,
    CustomerSupportResponse,
    HealthResponse,
    MarketingCampaignRequest,
    PVARequest,
    PVAResponse,
    TradingAnalysisRequest,
    TradingDecisionResponse,
)
from src.que_agents.error_trace.errorlogger import system_logger

AGENT_INITIALIZATION = "Agent Initialization"
CUSTOMER_SUPPORT_UNAVAILABLE = "Customer support agent not available"


class AgentManager:
    """Centralized agent management with better error handling"""

    def __init__(self):
        self.agents = {}
        self.agent_status = {
            "customer_support": False,
            "marketing": False,
            "personal_virtual_assistant": False,
            "financial_trading_bot": False,
        }
        self.fallback_agents = {}

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

            self.agents["customer_support"] = CustomerSupportAgent()
            self.agent_status["customer_support"] = True
            system_logger.info("Customer Support Agent initialized successfully")
        except Exception as e:
            system_logger.error(
                f"Failed to initialize Customer Support Agent: {str(e)}",
                additional_info={
                    "context": AGENT_INITIALIZATION,
                    "agent": "CustomerSupportAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["customer_support"] = None
            self._setup_customer_support_fallback()

    def _initialize_marketing_agent(self):
        """Initialize Marketing Agent"""
        try:
            from src.que_agents.agents.marketing_agent import MarketingAgent

            self.agents["marketing"] = MarketingAgent()
            self.agent_status["marketing"] = True
            system_logger.info("Marketing Agent initialized successfully")
        except Exception as e:
            system_logger.error(
                f"Failed to initialize Marketing Agent: {str(e)}",
                additional_info={
                    "context": AGENT_INITIALIZATION,
                    "agent": "MarketingAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["marketing"] = None
            self._setup_marketing_fallback()

    def _initialize_pva_agent(self):
        """Initialize Personal Virtual Assistant Agent with Groq configuration"""
        config = None
        try:
            with open("configs/agent_config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            system_logger.error(
                "Agent configuration file not found: configs/agent_config.yaml",
                additional_info={
                    "context": AGENT_INITIALIZATION,
                    "agent": "PersonalVirtualAssistantAgent",
                    "error_type": "FileNotFoundError",
                    "suggestion": "Create the agent configuration file",
                },
            )
            self.agents["personal_virtual_assistant"] = None
            self._setup_pva_fallback()
            return
        except Exception as e:
            system_logger.error(
                f"Error loading agent configuration: {str(e)}",
                additional_info={
                    "context": AGENT_INITIALIZATION,
                    "agent": "PersonalVirtualAssistantAgent",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.agents["personal_virtual_assistant"] = None
            self._setup_pva_fallback()
            return

        # Check if the agent config exists
        agent_config_key = "personal_virtual_assistant_agent"
        if config is None or agent_config_key not in config:
            system_logger.warning(
                f"Configuration key '{agent_config_key}' not found in agent_config.yaml"
            )
            # Try alternative key
            agent_config_key = "personal_virtual_assistant"
            if config is None or agent_config_key not in config:
                system_logger.error(
                    "No configuration found for PVA agent",
                    additional_info={
                        "context": AGENT_INITIALIZATION,
                        "agent": "PersonalVirtualAssistantAgent",
                        "error_type": "KeyError",
                        "suggestion": "Check agent_config.yaml for correct key names",
                    },
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

            self.agents["personal_virtual_assistant"] = test_agent
            self.agent_status["personal_virtual_assistant"] = True
            system_logger.info(
                f"Personal Virtual Assistant Agent initialized successfully with {config[agent_config_key].get('model_name', 'default')} model"
            )
        except Exception as e:
            system_logger.error(
                f"Failed to initialize Personal Virtual Assistant Agent: {str(e)}",
                additional_info={
                    "context": AGENT_INITIALIZATION,
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
            with open("configs/agent_config.yaml", "r") as f:
                config = yaml.safe_load(f)

            # Check if the agent config exists
            agent_config_key = "financial_trading_bot_agent"
            if agent_config_key not in config:
                # Try alternative key
                agent_config_key = "financial_trading_bot"
                if agent_config_key not in config:
                    system_logger.error(
                        "No configuration found for Trading Bot agent",
                        additional_info={
                            "context": AGENT_INITIALIZATION,
                            "agent": "FinancialTradingBotAgent",
                            "error_type": "KeyError",
                            "suggestion": "Check agent_config.yaml for correct key names",
                        },
                    )
                    raise KeyError("No configuration found for Trading Bot agent")

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

            self.agents["financial_trading_bot"] = test_agent
            self.agent_status["financial_trading_bot"] = True
            system_logger.info(
                f"Financial Trading Bot Agent initialized successfully with {config[agent_config_key].get('model_name', 'default')} model"
            )

        except Exception as e:
            system_logger.error(
                f"Failed to initialize Financial Trading Bot Agent: {str(e)}",
                additional_info={
                    "context": AGENT_INITIALIZATION,
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

    def get_agent(self, agent_name: str):
        """Get agent or fallback agent"""
        return self.agents.get(agent_name) or self.fallback_agents.get(agent_name)

    def is_agent_active(self, agent_name: str) -> str:
        """Check agent status"""
        if self.agents.get(agent_name):
            return "active"
        elif self.fallback_agents.get(agent_name):
            return "fallback"
        else:
            return "not_initialized"


class ConfigManager:
    """Configuration management with better error handling"""

    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load API configuration with fallback"""
        config_paths = [
            os.path.join(os.path.dirname(__file__), "../../../configs/api_config.yaml"),
            "configs/api_config.yaml",
            os.path.join(os.getcwd(), "configs/api_config.yaml"),
        ]

        for config_path in config_paths:
            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        system_logger.info(f"Configuration loaded from: {config_path}")
                        return config
            except Exception as e:
                system_logger.warning(
                    f"Failed to load config from {config_path}: {str(e)}"
                )
                continue

        # Fallback configuration
        system_logger.warning("Using fallback configuration")
        return {
            "api": {
                "title": "Agentic AI API",
                "description": "AI-powered customer support and marketing agents",
                "version": "1.0.0",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "host": "0.0.0.0",
                "port": 8000,
                "reload": True,
                "log_level": "info",
            },
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            },
            "authentication": {"api_token": "demo-token-123"},
        }


# Global instances
agent_manager = AgentManager()
config_manager = ConfigManager()
api_config = config_manager.load_config()


def get_system_metrics() -> Dict[str, Any]:
    """Collect comprehensive system metrics for health monitoring"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metrics = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent,
        }

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_metrics = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 2),
        }

        # Process metrics
        current_process = psutil.Process()
        process_metrics = {
            "memory_mb": round(current_process.memory_info().rss / (1024**2), 2),
            "cpu_percent": current_process.cpu_percent(),
            "num_threads": current_process.num_threads(),
            "create_time": current_process.create_time(),
        }

        # Agent-specific metrics
        agent_metrics = {
            "total_agents": len(agent_manager.agent_status),
            "active_agents": sum(
                1 for status in agent_manager.agent_status.values() if status
            ),
            "fallback_agents": len(agent_manager.fallback_agents),
            "agent_health_score": calculate_agent_health_score(),
        }

        return {
            "cpu": {"percent_used": cpu_percent, "core_count": cpu_count},
            "memory": memory_metrics,
            "disk": disk_metrics,
            "process": process_metrics,
            "agents": agent_metrics,
            "uptime_seconds": time.time() - psutil.boot_time(),
            "load_average": (
                list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None
            ),
        }

    except Exception as e:
        system_logger.error(f"Error collecting system metrics: {str(e)}")
        return {"error": "Failed to collect system metrics", "message": str(e)}


def calculate_agent_health_score() -> float:
    """Calculate overall agent health score (0.0 to 1.0)"""
    total_agents = len(agent_manager.agent_status)
    if total_agents == 0:
        return 0.0

    active_agents = sum(1 for status in agent_manager.agent_status.values() if status)
    fallback_agents = len(agent_manager.fallback_agents)

    # Active agents get full score, fallback agents get partial score
    score = (active_agents + (fallback_agents * 0.5)) / total_agents
    return round(min(score, 1.0), 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    system_logger.info("Starting Agentic AI API...")
    agent_manager.initialize_agents()
    yield
    # Shutdown
    system_logger.info("Shutting down Agentic AI API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=api_config["api"]["title"],
    description=api_config["api"]["description"],
    version=api_config["api"]["version"],
    docs_url=api_config["api"]["docs_url"],
    redoc_url=api_config["api"]["redoc_url"],
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config["cors"]["allow_origins"],
    allow_credentials=api_config["cors"]["allow_credentials"],
    allow_methods=api_config["cors"]["allow_methods"],
    allow_headers=api_config["cors"]["allow_headers"],
)

# Security
security = HTTPBearer()

# Constants
INVALID_AUTH_TOKEN_MSG = "Invalid authentication token"


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    expected_token = api_config["authentication"]["api_token"]

    if credentials.credentials != expected_token:
        system_logger.error(
            INVALID_AUTH_TOKEN_MSG,
            additional_info={"context": "Authentication"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_AUTH_TOKEN_MSG,
        )
    return credentials.credentials


# Enhanced health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with detailed agent status and system metrics"""
    agent_statuses = {}

    for agent_name in agent_manager.agent_status.keys():
        agent_statuses[agent_name] = agent_manager.is_agent_active(agent_name)

    # Determine overall status
    active_count = sum(1 for status in agent_statuses.values() if status == "active")
    fallback_count = sum(
        1 for status in agent_statuses.values() if status == "fallback"
    )
    total_agents = len(agent_statuses)

    if active_count == total_agents:
        overall_status = "healthy"
    elif active_count + fallback_count >= total_agents * 0.5:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    # Collect system metrics
    system_metrics = get_system_metrics()

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        agents=agent_statuses,
        system_metrics=system_metrics,
    )


# Additional endpoint for detailed system monitoring
@app.get("/api/v1/system/metrics")
async def get_detailed_metrics(token: str = Depends(verify_token)):
    """Get detailed system metrics for monitoring and alerting"""
    metrics = get_system_metrics()

    # Add additional monitoring data
    metrics.update(
        {
            "api_info": {
                "version": api_config["api"]["version"],
                "title": api_config["api"]["title"],
                "environment": os.getenv("ENVIRONMENT", "development"),
            },
            "agent_details": {
                name: {
                    "status": agent_manager.is_agent_active(name),
                    "initialized": agent_manager.agent_status.get(name, False),
                    "has_fallback": name in agent_manager.fallback_agents,
                }
                for name in agent_manager.agent_status.keys()
            },
            "configuration": {
                "cors_enabled": bool(api_config.get("cors")),
                "auth_enabled": bool(api_config.get("authentication")),
                "docs_enabled": api_config["api"].get("docs_url") is not None,
            },
        }
    )

    return metrics


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic AI API is running",
        "version": api_config["api"]["version"],
        "status": "healthy",
    }


# Customer Support endpoints
@app.post("/api/v1/customer-support/chat", response_model=CustomerSupportResponse)
async def customer_support_chat(
    request: CustomerSupportRequest, token: str = Depends(verify_token)
):
    """Handle customer support chat request"""
    try:
        agent = agent_manager.get_agent("customer_support")
        if not agent:
            raise HTTPException(status_code=503, detail=CUSTOMER_SUPPORT_UNAVAILABLE)

        result = agent.handle_customer_request_enhanced(
            customer_id=request.customer_id, message=request.message
        )

        return CustomerSupportResponse(
            response=result["response"],
            confidence=result["confidence"],
            escalate=result["escalate"],
            suggested_actions=result["suggested_actions"],
            knowledge_sources=result["knowledge_sources"],
            sentiment=result["sentiment"],
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        system_logger.error(
            f"Error handling customer support chat: {e}",
            additional_info={
                "context": "Customer Support Chat",
                "customer_id": request.customer_id,
            },
            exc_info=True,
        )
        # Return fallback response
        return CustomerSupportResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please contact our support team directly for immediate assistance.",
            confidence=0.5,
            escalate=True,
            suggested_actions=["contact_human_support"],
            knowledge_sources=["error_handling"],
            sentiment="neutral",
            timestamp=datetime.now().isoformat(),
        )


# Add this endpoint to your main.py file


@app.get("/api/v1/customer-support/customer/{customer_id}")
async def get_customer_context(customer_id: int, token: str = Depends(verify_token)):
    """Get customer context and information"""
    try:
        agent = agent_manager.get_agent("customer_support")
        if not agent:
            raise HTTPException(status_code=503, detail=CUSTOMER_SUPPORT_UNAVAILABLE)

        # Get customer context using the agent's method
        customer_context = agent.get_customer_context(customer_id)

        if not customer_context:
            raise HTTPException(
                status_code=404, detail=f"Customer with ID {customer_id} not found"
            )

        return {
            "customer_id": customer_context.customer_id,
            "customer_name": customer_context.name,
            "email": customer_context.email,
            "support_tier": customer_context.tier,
            "company": customer_context.company,
            "satisfaction_score": customer_context.satisfaction_score,
            "lifetime_value": customer_context.lifetime_value,
            "risk_score": customer_context.risk_score,
            "recent_interactions": customer_context.recent_interactions,
            "open_tickets": customer_context.open_tickets,
            "preferences": customer_context.preferences,
            "purchase_history": customer_context.purchase_history,
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        system_logger.error(
            f"Error getting customer context: {str(e)}",
            additional_info={
                "context": "Get Customer Context",
                "customer_id": customer_id,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving customer context: {str(e)}"
        )


# Add customer insights endpoint as well
@app.get("/api/v1/customer-support/customer/{customer_id}/insights")
async def get_customer_insights(customer_id: int, token: str = Depends(verify_token)):
    """Get comprehensive customer insights"""
    try:
        agent = agent_manager.get_agent("customer_support")
        if not agent:
            raise HTTPException(status_code=503, detail=CUSTOMER_SUPPORT_UNAVAILABLE)

        insights = agent.get_customer_insights(customer_id)

        if "error" in insights:
            raise HTTPException(status_code=404, detail=insights["error"])

        return insights

    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(
            f"Error getting customer insights: {str(e)}",
            additional_info={
                "context": "Get Customer Insights",
                "customer_id": customer_id,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving customer insights: {str(e)}"
        )


# Add other missing endpoints that the HTML is trying to call


@app.get("/api/v1/pva/user/{user_id}/reminders")
async def get_user_reminders(user_id: str, token: str = Depends(verify_token)):
    """Get user reminders from PVA"""
    try:
        agent = agent_manager.get_agent("personal_virtual_assistant")
        if not agent:
            return {"reminders": [], "total": 0, "message": "PVA agent not available"}

        # Check if agent has the method
        if hasattr(agent, "get_user_reminders"):
            reminders = agent.get_user_reminders(user_id)
            return reminders
        else:
            # Fallback response
            return {
                "reminders": [
                    {
                        "id": 1,
                        "title": "Sample Reminder",
                        "target_time": datetime.now().isoformat(),
                        "status": "active",
                    }
                ],
                "total": 1,
                "message": "Using fallback reminder data",
            }

    except Exception as e:
        system_logger.error(f"Error getting user reminders: {str(e)}")
        return {"reminders": [], "total": 0, "error": str(e)}


@app.get("/api/v1/pva/user/{user_id}/devices")
async def get_user_devices(user_id: str, token: str = Depends(verify_token)):
    """Get user smart devices from PVA"""
    try:
        agent = agent_manager.get_agent("personal_virtual_assistant")
        if not agent:
            return {"devices": [], "total": 0, "message": "PVA agent not available"}

        # Check if agent has the method
        if hasattr(agent, "get_user_devices"):
            devices = agent.get_user_devices(user_id)
            return devices
        else:
            # Fallback response
            return {
                "devices": [
                    {
                        "id": 1,
                        "name": "Smart Light",
                        "type": "lighting",
                        "status": "on",
                    },
                    {
                        "id": 2,
                        "name": "Thermostat",
                        "type": "climate",
                        "status": "auto",
                    },
                ],
                "total": 2,
                "message": "Using fallback device data",
            }

    except Exception as e:
        system_logger.error(f"Error getting user devices: {str(e)}")
        return {"devices": [], "total": 0, "error": str(e)}


# Add analytics endpoint
@app.get("/api/v1/analytics")
async def get_analytics(token: str = Depends(verify_token)):
    """Get analytics data for dashboard"""
    try:
        # Get system metrics
        get_system_metrics()

        return {
            "system_metrics": {
                "total-customers": 150,  # Mock data - replace with real queries
                "total-campaigns": 25,
                "knowledge-base-docs": 500,
                "api-status": "healthy",
                "customer-support-status": "active",
                "marketing-status": "active",
                "pva-status": "active",
                "trading-bot-status": "active",
            },
            "recent_interactions": [
                {
                    "agent_name": "Customer Support",
                    "timestamp": datetime.now().isoformat(),
                    "interaction_summary": "Password reset assistance",
                },
                {
                    "agent_name": "Marketing Agent",
                    "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                    "interaction_summary": "Campaign creation for tech products",
                },
                {
                    "agent_name": "PVA",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "interaction_summary": "Weather inquiry and reminders",
                },
            ],
        }

    except Exception as e:
        system_logger.error(f"Error getting analytics: {str(e)}")
        return {"error": str(e)}


# Add content generation endpoint for marketing
@app.post("/api/v1/marketing/content/generate")
async def generate_marketing_content(request: dict, token: str = Depends(verify_token)):
    """Generate marketing content"""
    try:
        agent = agent_manager.get_agent("marketing")
        if not agent:
            raise HTTPException(status_code=503, detail="Marketing agent not available")

        # Generate content based on request
        return {
            "email_subject": f"Exclusive Offer for {request.get('target_audience', 'Valued Customers')}",
            "email_body": f"We're excited to announce our latest {request.get('campaign_type', 'promotion')} designed specifically for {request.get('target_audience', 'you')}. Don't miss out on this opportunity!",
            "social_media_post": f"🚀 New {request.get('campaign_type', 'campaign')} alert! Perfect for {request.get('target_audience', 'professionals')}. Check it out! #Innovation #Growth",
            "status": "generated",
        }

    except Exception as e:
        system_logger.error(f"Error generating content: {str(e)}")
        return {"error": str(e), "status": "failed"}


# Add trading endpoints
@app.post("/api/v1/trading/cycle")
async def run_trading_cycle(token: str = Depends(verify_token)):
    """Run trading cycle"""
    try:
        agent = agent_manager.get_agent("financial_trading_bot")
        if not agent:
            raise HTTPException(
                status_code=503, detail="Financial Trading Bot not available"
            )

        # Use fallback or agent method
        if hasattr(agent, "run_trading_cycle"):
            result = agent.run_trading_cycle(["AAPL", "GOOGL", "MSFT"])
        else:
            result = {
                "timestamp": datetime.now().isoformat(),
                "symbols_analyzed": 3,
                "trades_executed": 0,
                "decisions": [
                    {
                        "symbol": "AAPL",
                        "action": "hold",
                        "confidence": 0.7,
                        "executed": False,
                    },
                    {
                        "symbol": "GOOGL",
                        "action": "hold",
                        "confidence": 0.6,
                        "executed": False,
                    },
                    {
                        "symbol": "MSFT",
                        "action": "hold",
                        "confidence": 0.8,
                        "executed": False,
                    },
                ],
                "portfolio_status": {"total_value": 10000.0, "cash_balance": 5000.0},
                "average_confidence": 0.7,
            }

        return result

    except Exception as e:
        system_logger.error(f"Error in trading cycle: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trading/portfolio")
async def get_portfolio_status(token: str = Depends(verify_token)):
    """Get portfolio status"""
    try:
        agent = agent_manager.get_agent("financial_trading_bot")
        if not agent:
            return {
                "portfolio_value": 10000.0,
                "cash_balance": 5000.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "holdings": {},
                "performance_metrics": {},
            }

        if hasattr(agent, "get_portfolio_status"):
            return agent.get_portfolio_status()
        else:
            return {
                "portfolio_value": 10000.0,
                "cash_balance": 5000.0,
                "unrealized_pnl": 250.0,
                "realized_pnl": 150.0,
                "holdings": {
                    "AAPL": {"shares": 10, "value": 1500.0},
                    "GOOGL": {"shares": 5, "value": 1250.0},
                },
                "performance_metrics": {
                    "total_return": 4.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -2.5,
                },
            }

    except Exception as e:
        system_logger.error(f"Error getting portfolio status: {str(e)}")
        return {"error": str(e)}


@app.get("/api/v1/trading/market/{symbol}")
async def get_market_data(symbol: str, token: str = Depends(verify_token)):
    """Get market data for symbol"""
    try:
        agent = agent_manager.get_agent("financial_trading_bot")
        if not agent:
            # Return mock data
            return {
                "symbol": symbol,
                "current_price": 150.0,
                "change_24h": 2.5,
                "volume": 1000000,
                "rsi": 65.0,
                "macd": 0.5,
                "moving_avg_20": 148.0,
                "moving_avg_50": 145.0,
                "volatility": 0.2,
                "market_sentiment": "bullish",
            }

        if hasattr(agent, "get_market_data"):
            return agent.get_market_data(symbol)
        else:
            return {
                "symbol": symbol,
                "current_price": 150.0,
                "change_24h": 2.5,
                "volume": 1000000,
                "rsi": 65.0,
                "macd": 0.5,
                "moving_avg_20": 148.0,
                "moving_avg_50": 145.0,
                "volatility": 0.2,
                "market_sentiment": "neutral",
            }

    except Exception as e:
        system_logger.error(f"Error getting market data: {str(e)}")
        return {"error": str(e)}


# Marketing endpoints
@app.post("/api/v1/marketing/campaign/create")
def create_marketing_campaign(
    request: MarketingCampaignRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Create a new marketing campaign"""
    try:
        agent = agent_manager.get_agent("marketing")
        if not agent:
            raise HTTPException(status_code=503, detail="Marketing agent not available")

        # Convert Pydantic model to dict
        request_dict = {
            "campaign_type": request.campaign_type,
            "target_audience": request.target_audience,
            "budget": request.budget,
            "duration_days": request.duration_days,
            "goals": request.goals,
            "channels": request.channels,
            "content_requirements": request.content_requirements,
            "industry": request.industry,
            "brand_voice": request.brand_voice,
        }

        result = agent.create_marketing_campaign(request_dict)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to create campaign: {str(e)}",
            additional_info={
                "context": "Create Marketing Campaign",
                "request": request.dict(),
            },
            exc_info=True,
        )
        return {"error": f"Failed to create campaign: {str(e)}", "status": "failed"}


# Personal Virtual Assistant endpoints
@app.post("/api/v1/pva/chat", response_model=PVAResponse)
async def pva_chat(request: PVARequest, token: str = Depends(verify_token)):
    """Handle Personal Virtual Assistant chat request"""
    try:
        agent = agent_manager.get_agent("personal_virtual_assistant")
        if not agent:
            raise HTTPException(
                status_code=503, detail="Personal Virtual Assistant not available"
            )

        result = agent.handle_user_request(
            user_id=request.user_id,
            user_message=request.message,
            session_id=request.session_id or "",
        )

        return PVAResponse(
            response=result["response"],
            intent=result["intent"],
            entities=result["entities"],
            confidence=result["confidence"],
            actions_taken=result["actions_taken"],
            suggestions=result["suggestions"],
            timestamp=result["timestamp"],
        )
    except Exception as e:
        system_logger.error(
            f"Error in PVA chat: {str(e)}",
            additional_info={
                "context": "PVA Chat",
                "user_id": request.user_id,
            },
            exc_info=True,
        )

        return PVAResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            intent="error",
            entities={},
            confidence=0.0,
            actions_taken=["error_handling"],
            suggestions=["Try again later"],
            timestamp=datetime.now().isoformat(),
        )


# Financial Trading Bot endpoints
@app.post("/api/v1/trading/analyze", response_model=TradingDecisionResponse)
async def analyze_and_decide(
    request: TradingAnalysisRequest, token: str = Depends(verify_token)
):
    """Analyze market and make trading decision"""
    try:
        agent = agent_manager.get_agent("financial_trading_bot")
        if not agent:
            raise HTTPException(
                status_code=503, detail="Financial Trading Bot not available"
            )

        decision = agent.make_trading_decision(
            symbol=request.symbol, strategy_type=request.strategy_type
        )

        return TradingDecisionResponse(
            action=decision.action,
            symbol=decision.symbol,
            quantity=decision.quantity,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            risk_score=decision.risk_score,
            expected_return=decision.expected_return,
        )
    except Exception as e:
        system_logger.error(
            f"Error in trading analysis: {str(e)}",
            additional_info={
                "context": "Trading Analysis",
                "symbol": request.symbol,
            },
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trading decision: {str(e)}",
        )


@app.get("/api/v1/diagnostics")
async def get_diagnostics():
    """Get detailed diagnostic information for troubleshooting"""
    import sys
    from pathlib import Path

    diagnostics = {
        "python_version": sys.version,
        "current_working_directory": os.getcwd(),
        "python_path": sys.path[:5],
        "environment_variables": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("OPENAI_", "AZURE_", "API_", "PYTHONPATH"))
        },
        "agents_initialization": agent_manager.agent_status,
        "agents_current_status": {
            name: agent_manager.is_agent_active(name)
            for name in agent_manager.agent_status.keys()
        },
        "config_loaded": bool(api_config),
        "file_structure": {},
    }

    # Check if agent files exist
    base_path = Path(__file__).parent.parent
    agent_files = [
        "agents/customer_support_agent.py",
        "agents/marketing_agent.py",
        "agents/personal_virtual_assistant_agent.py",
        "agents/financial_trading_bot_agent.py",
    ]

    for agent_file in agent_files:
        file_path = base_path / agent_file
        diagnostics["file_structure"][agent_file] = {
            "exists": file_path.exists(),
            "path": str(file_path),
            "is_file": file_path.is_file() if file_path.exists() else False,
        }

    return diagnostics


@app.get("/debug")
async def debug_info():
    """Debug information"""
    return {
        "config_loaded": bool(api_config),
        "agents_loaded": {
            name: agent_manager.get_agent(name) is not None
            for name in agent_manager.agent_status.keys()
        },
        "agents_status": {
            name: agent_manager.is_agent_active(name)
            for name in agent_manager.agent_status.keys()
        },
        "routes": [
            getattr(route, "path", None)
            for route in app.routes
            if getattr(route, "path", None) is not None
        ],
        "fallback_agents_active": list(agent_manager.fallback_agents.keys()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=api_config["api"]["host"],
        port=api_config["api"]["port"],
        reload=api_config["api"]["reload"],
    )
