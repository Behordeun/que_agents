# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements the main API for the Agentic AI system

import os
import traceback
from datetime import datetime
from typing import List, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.que_agents.agents.customer_support_agent import CustomerSupportAgent
from src.que_agents.agents.financial_trading_bot_agent import FinancialTradingBotAgent
from src.que_agents.agents.marketing_agent import MarketingAgent
from src.que_agents.agents.personal_virtual_assistant_agent import (
    PersonalVirtualAssistantAgent,
)
from src.que_agents.core.schemas import (
    ContentGenerationRequest,
    CustomerSupportRequest,
    CustomerSupportResponse,
    HealthResponse,
    MarketingCampaignRequest,
    PVARequest,
    PVAResponse,
    TradingAnalysisRequest,
    TradingCycleRequest,
    TradingCycleResponse,
    TradingDecisionResponse,
)
from src.que_agents.error_trace.errorlogger import system_logger

system_logger.info("Starting Agentic AI API...")

# Load API configuration - fix the path
config_path = os.path.join(
    os.path.dirname(__file__), "../../../configs/api_config.yaml"
)
if not os.path.exists(config_path):
    # Alternative path
    config_path = "configs/api_config.yaml"

try:
    with open(config_path, "r") as f:
        api_config = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback configuration
    print(f"Warning: Config file not found at {config_path}. Using fallback config.")
    api_config = {
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

# Initialize FastAPI app
app = FastAPI(
    title=api_config["api"]["title"],
    description=api_config["api"]["description"],
    version=api_config["api"]["version"],
    docs_url=api_config["api"]["docs_url"],
    redoc_url=api_config["api"]["redoc_url"],
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

# Initialize agents with error handling
try:
    customer_support_agent = CustomerSupportAgent()
    marketing_agent = MarketingAgent()
    pva_agent = PersonalVirtualAssistantAgent()
    trading_bot_agent = FinancialTradingBotAgent()
except Exception as e:
    system_logger.error(
        str(e),
        additional_info={
            "context": "Error initializing agents",
            "message": "Error initializing agents. Please check your configuration and agent implementations.",
        },
        exc_info=True,
    )
    customer_support_agent = None
    marketing_agent = None
    pva_agent = None
    trading_bot_agent = None

# Constants
MARKETING_AGENT_NOT_AVAILABLE = "Marketing agent not available"
CREATE_MARKETING_CAMPAIGN_CONTEXT = "Create Marketing Campaign"


# Authentication error constant
INVALID_AUTH_TOKEN_MSG = "Invalid authentication token"


# Authentication dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)"""
    expected_token = api_config["authentication"]["api_token"]

    if credentials.credentials != expected_token:
        system_logger.error(
            INVALID_AUTH_TOKEN_MSG,
            additional_info={
                "context": "Authentication",
                "message": INVALID_AUTH_TOKEN_MSG,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_AUTH_TOKEN_MSG,
        )
    return credentials.credentials


# Health check endpoint - MUST be defined
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agents={
            "customer_support": "active" if customer_support_agent else "error",
            "marketing": "active" if marketing_agent else "error",
            "personal_virtual_assistant": "active" if pva_agent else "error",
            "financial_trading_bot": "active" if trading_bot_agent else "error",
        },
    )


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic AI API is running",
        "version": api_config["api"]["version"],
    }


# Customer Support endpoints
@app.post("/api/v1/customer-support/chat", response_model=CustomerSupportResponse)
async def customer_support_chat(
    request: CustomerSupportRequest, token: str = Depends(verify_token)
):
    """Handle customer support chat request"""
    try:
        if not customer_support_agent:
            # Return mock response if agent not available
            return CustomerSupportResponse(
                response="I understand you're having login issues. Let me help you with that. First, please try resetting your password using the 'Forgot Password' link on the login page.",
                confidence=0.85,
                escalate=False,
                suggested_actions=[
                    "reset_password",
                    "clear_browser_cache",
                    "contact_support",
                ],
                knowledge_sources=["login_troubleshooting", "password_reset_guide"],
                sentiment="neutral",
                timestamp=datetime.now().isoformat(),
            )

        # Call the actual agent
        result = customer_support_agent.handle_customer_request_enhanced(
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
                "message": request.message,
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


@app.get("/api/v1/customer-support/customer/{customer_id}")
async def get_customer_context(customer_id: int, token: str = Depends(verify_token)):
    """Get customer context information"""
    mock_customers = {
        1: {
            "name": "John Smith",
            "email": "john@example.com",
            "tier": "Premium",
            "company": "TechCorp",
        },
        2: {
            "name": "Sarah Johnson",
            "email": "sarah@example.com",
            "tier": "Standard",
            "company": "StartupInc",
        },
        3: {
            "name": "Mike Wilson",
            "email": "mike@example.com",
            "tier": "Enterprise",
            "company": "BigCorp",
        },
        4: {
            "name": "Emily Davis",
            "email": "emily@example.com",
            "tier": "Standard",
            "company": "MidSize Co",
        },
        5: {
            "name": "David Brown",
            "email": "david@example.com",
            "tier": "Premium",
            "company": "GrowthCorp",
        },
    }

    if customer_id not in mock_customers:
        system_logger.error(
            f"Customer {customer_id} not found in mock data",
            additional_info={
                "context": "Get Customer Context",
                "customer_id": customer_id,
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found",
        )

    customer = mock_customers[customer_id]
    return {**customer, "open_tickets": [], "recent_interactions": []}


# Marketing endpoints with corrected method calls
@app.post("/api/v1/marketing/campaign/create")
def create_marketing_campaign(
    request: MarketingCampaignRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Create a new marketing campaign"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail="Marketing agent not available")

        # Convert Pydantic model to dict - this is the issue
        request_dict = {
            "campaign_type": request.campaign_type,
            "target_audience": request.target_audience,
            "budget": request.budget,
            "duration_days": request.duration_days,
            "goals": request.goals,
            "channels": request.channels,
            "content_requirements": request.content_requirements,  # These are enum values, not strings
            "industry": request.industry,
            "brand_voice": request.brand_voice,
        }

        result = marketing_agent.create_marketing_campaign(request_dict)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to create campaign: {str(e)}",
            additional_info={
                "context": CREATE_MARKETING_CAMPAIGN_CONTEXT,
                "request": request.dict(),
            },
        )
        return {"error": f"Failed to create campaign: {str(e)}", "status": "failed"}


@app.post("/api/v1/marketing/content/generate")
def generate_marketing_content(
    request: ContentGenerationRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        # Convert Pydantic model to dict
        request_dict = {
            "platform": request.platform,
            "content_type": request.content_type,
            "campaign_theme": request.campaign_theme,
            "target_audience": request.target_audience,
            "key_messages": request.key_messages,
            "brand_voice": request.brand_voice,
            "tone": request.tone,
            "include_hashtags": request.include_hashtags,
        }

        result = marketing_agent.generate_marketing_content(request_dict)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to generate content: {str(e)}",
            additional_info={
                "context": "Generate Marketing Content",
                "request": request.dict(),
            },
        )
        return {
            "error": f"Failed to generate content: {str(e)}",
            "status": "failed",
            "content": "Unable to generate content at this time.",
        }


@app.get("/api/v1/marketing/campaign/{campaign_id}/analyze")
def analyze_marketing_campaign(
    campaign_id: int, _: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Analyze marketing campaign performance"""
    try:
        if not marketing_agent:
            system_logger.error(
                f"{MARKETING_AGENT_NOT_AVAILABLE}",
                additional_info={
                    "context": "Analyze Marketing Campaign",
                    "campaign_id": campaign_id,
                },
            )
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.analyze_campaign_performance(campaign_id)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to analyze campaign: {str(e)}",
            additional_info={
                "context": "Analyze Marketing Campaign",
                "campaign_id": campaign_id,
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to analyze campaign: {str(e)}",
            "status": "failed",
            "campaign_id": campaign_id,
        }


@app.post("/api/v1/marketing/campaign/{campaign_id}/optimize")
def optimize_marketing_campaign(
    campaign_id: int,
    optimization_goals: Optional[List[str]] = None,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Optimize marketing campaign"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.optimize_campaign(campaign_id, optimization_goals)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to optimize campaign: {str(e)}",
            additional_info={
                "context": "Optimize Marketing Campaign",
                "campaign_id": campaign_id,
                "optimization_goals": optimization_goals,
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to optimize campaign: {str(e)}",
            "status": "failed",
            "campaign_id": campaign_id,
        }


@app.get("/api/v1/marketing/campaign/{campaign_id}/insights")
def get_marketing_campaign_insights(
    campaign_id: int, _: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get marketing campaign insights"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.get_campaign_insights(campaign_id)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to get campaign insights: {str(e)}",
            additional_info={
                "context": "GET MARKETING CAMPAIGN INSIGHTS",
            },
        )
        return {
            "error": f"Failed to get campaign insights: {str(e)}",
            "status": "failed",
            "campaign_id": campaign_id,
        }


@app.get("/api/v1/marketing/audience/analyze")
def analyze_target_audience(
    target_audience: str,
    industry: Optional[str] = None,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Analyze target audience"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.get_audience_analysis(target_audience, industry)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to analyze audience: {str(e)}",
            additional_info={
                "context": "Analyze Target Audience",
                "target_audience": target_audience,
                "industry": industry,
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to analyze audience: {str(e)}",
            "status": "failed",
            "target_audience": target_audience,
        }


@app.get("/api/v1/marketing/market/intelligence")
def get_market_intelligence(
    campaign_type: str,
    industry: Optional[str] = None,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Get market intelligence"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.get_market_intelligence(campaign_type, industry)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to get market intelligence: {str(e)}",
            additional_info={
                "context": "Get Market Intelligence",
                "campaign_type": campaign_type,
                "industry": industry,
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to get market intelligence: {str(e)}",
            "status": "failed",
            "campaign_type": campaign_type,
        }


@app.get("/api/v1/marketing/content/suggestions")
def get_content_suggestions(
    platform: str,
    industry: Optional[str] = None,
    content_type: str = "social_media",
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Get content suggestions for platform"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.get_content_suggestions(
            platform, industry, content_type
        )
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to get content suggestions: {str(e)}",
            additional_info={
                "context": "Get Content Suggestions",
                "platform": platform,
                "industry": industry,
                "content_type": content_type,
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to get content suggestions: {str(e)}",
            "status": "failed",
            "platform": platform,
        }


@app.get("/api/v1/marketing/templates")
def get_campaign_templates(
    campaign_type: str,
    industry: Optional[str] = None,
    _: HTTPAuthorizationCredentials = Depends(verify_token),
):
    """Get campaign templates"""
    try:
        if not marketing_agent:
            raise HTTPException(status_code=503, detail=MARKETING_AGENT_NOT_AVAILABLE)

        result = marketing_agent.get_campaign_templates(campaign_type, industry)
        return result

    except Exception as e:
        system_logger.error(
            f"Failed to get campaign templates: {str(e)}",
            additional_info={
                "context": "Get Campaign Templates",
                "campaign_type": campaign_type,
                "industry": industry,
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to get campaign templates: {str(e)}",
            "status": "failed",
            "campaign_type": campaign_type,
        }


# Add a debug endpoint to show valid enum values
@app.get("/api/v1/marketing/enums")
def get_marketing_enums():
    """Get valid enum values for marketing endpoints"""
    try:
        from src.que_agents.core.schemas import CampaignType, ContentType

        return {
            "campaign_types": [ct.value for ct in CampaignType],
            "content_types": [ct.value for ct in ContentType],
            "platforms": [
                "twitter",
                "linkedin",
                "facebook",
                "instagram",
                "email",
                "youtube",
                "tiktok",
            ],
            "brand_voices": [
                "professional",
                "casual",
                "friendly",
                "authoritative",
                "playful",
            ],
            "industries": [
                "technology",
                "healthcare",
                "finance",
                "retail",
                "education",
                "manufacturing",
            ],
        }
    except Exception as e:
        system_logger.error(
            f"Failed to get enum values: {str(e)}",
            additional_info={
                "context": "Get Marketing Enums",
            },
            exc_info=True,
        )
        return {
            "error": f"Failed to get enum values: {str(e)}",
            "campaign_types": ["brand_awareness", "lead_generation", "product_launch"],
            "content_types": ["social_media", "email", "blog_post", "video"],
        }


# Personal Virtual Assistant endpoints
@app.post("/api/v1/pva/chat", response_model=PVAResponse)
async def pva_chat(request: PVARequest, token: str = Depends(verify_token)):
    """Handle Personal Virtual Assistant chat request"""
    try:
        if not pva_agent:
            # Return mock response if agent not available
            return PVAResponse(
                response="Hello! I'm your personal assistant. I can help you with weather, reminders, device control, and general questions. How can I assist you today?",
                intent="greeting",
                entities={},
                confidence=0.9,
                actions_taken=["greeted_user"],
                suggestions=[
                    "Ask about weather",
                    "Set a reminder",
                    "Control smart devices",
                ],
                timestamp=datetime.now().isoformat(),
            )

        # Call the actual agent
        result = pva_agent.handle_user_request(
            user_id=request.user_id,
            user_message=request.message,
            session_id=request.session_id if request.session_id is not None else "",
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
                "session_id": request.session_id,
            },
            exc_info=True,
        )

        traceback.print_exc()
        return PVAResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            intent="error",
            entities={},
            confidence=0.0,
            actions_taken=["error_handling"],
            suggestions=["Try again later"],
            timestamp=datetime.now().isoformat(),
        )


@app.get("/api/v1/pva/user/{user_id}/reminders")
async def get_user_reminders(user_id: str, token: str = Depends(verify_token)):
    """Get user's active reminders"""
    try:
        if not pva_agent:
            return {"reminders": [], "total": 0}

        user_context = pva_agent.get_user_context(user_id)
        if user_context:
            return {
                "reminders": user_context.active_reminders,
                "total": len(user_context.active_reminders),
            }
        else:
            return {"reminders": [], "total": 0}
    except Exception as e:
        system_logger.error(
            f"Error getting reminders: {str(e)}",
            additional_info={
                "context": "Get User Reminders",
                "user_id": user_id,
            },
            exc_info=True,
        )
        return {"reminders": [], "total": 0, "error": str(e)}


@app.get("/api/v1/pva/user/{user_id}/devices")
async def get_user_devices(user_id: str, token: str = Depends(verify_token)):
    """Get user's smart devices"""
    try:
        if not pva_agent:
            return {"devices": [], "total": 0}

        user_context = pva_agent.get_user_context(user_id)
        if user_context:
            return {
                "devices": user_context.smart_devices,
                "total": len(user_context.smart_devices),
            }
        else:
            return {"devices": [], "total": 0}
    except Exception as e:
        system_logger.error(
            f"Error getting devices: {str(e)}",
            additional_info={
                "context": "Get User Devices",
                "user_id": user_id,
            },
            exc_info=True,
        )
        return {"devices": [], "total": 0, "error": str(e)}


# Financial Trading Bot endpoints
@app.post("/api/v1/trading/analyze", response_model=TradingDecisionResponse)
async def analyze_and_decide(
    request: TradingAnalysisRequest, token: str = Depends(verify_token)
):
    """Analyze market and make trading decision"""
    try:
        if not trading_bot_agent:
            # Return mock response if agent not available
            return TradingDecisionResponse(
                action="hold",
                symbol=request.symbol,
                quantity=0.0,
                confidence=0.7,
                reasoning=f"Market analysis for {request.symbol} suggests holding position due to mixed signals.",
                risk_score=0.5,
                expected_return=0.02,
            )

        # Call the actual agent
        decision = trading_bot_agent.make_trading_decision(
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
                "strategy_type": request.strategy_type,
            },
            exc_info=True,
        )

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trading decision: {str(e)}",
        )


@app.post("/api/v1/trading/cycle", response_model=TradingCycleResponse)
async def run_trading_cycle(
    request: TradingCycleRequest, token: str = Depends(verify_token)
):
    """Run a complete trading cycle"""
    try:
        if not trading_bot_agent:
            # Return mock response if agent not available
            symbols = request.symbols or ["AAPL", "GOOGL", "MSFT"]
            return TradingCycleResponse(
                timestamp=datetime.now().isoformat(),
                symbols_analyzed=len(symbols),
                trades_executed=1,
                decisions=[
                    {
                        "symbol": symbol,
                        "action": "hold",
                        "confidence": 0.6,
                        "executed": False,
                    }
                    for symbol in symbols
                ],
                portfolio_status={"total_value": 10000.0, "cash_balance": 5000.0},
                average_confidence=0.6,
            )

        # Call the actual agent
        symbols = (
            request.symbols
            if request.symbols is not None
            else ["AAPL", "GOOGL", "MSFT"]
        )
        result = trading_bot_agent.run_trading_cycle(symbols=symbols)

        return TradingCycleResponse(
            timestamp=result["timestamp"],
            symbols_analyzed=result["symbols_analyzed"],
            trades_executed=result["trades_executed"],
            decisions=result["decisions"],
            portfolio_status=result["portfolio_status"],
            average_confidence=result["average_confidence"],
        )
    except Exception as e:
        system_logger.error(
            f"Error in trading cycle: {str(e)}",
            additional_info={
                "context": "Trading Cycle",
                "symbols": request.symbols,
            },
            exc_info=True,
        )

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running trading cycle: {str(e)}",
        )


@app.get("/api/v1/trading/portfolio")
async def get_portfolio_status(token: str = Depends(verify_token)):
    """Get current portfolio status"""
    try:
        if not trading_bot_agent:
            return {
                "portfolio_value": 10000.0,
                "cash_balance": 5000.0,
                "total_return": 0.0,
                "holdings": {},
                "performance_metrics": {},
            }

        portfolio_status = trading_bot_agent.get_portfolio_status()
        return {
            "portfolio_value": portfolio_status.total_value,
            "cash_balance": portfolio_status.cash_balance,
            "holdings": portfolio_status.holdings,
            "performance_metrics": portfolio_status.performance_metrics,
            "unrealized_pnl": portfolio_status.unrealized_pnl,
            "realized_pnl": portfolio_status.realized_pnl,
        }
    except Exception as e:
        system_logger.error(
            f"Error getting portfolio: {str(e)}",
            additional_info={
                "context": "Get Portfolio Status",
            },
            exc_info=True,
        )
        return {"error": str(e)}


@app.get("/api/v1/trading/performance")
async def get_performance_report(token: str = Depends(verify_token)):
    """Get trading performance report"""
    try:
        if not trading_bot_agent:
            return {
                "portfolio_value": 10000.0,
                "total_return": 0.0,
                "trade_statistics": {"total_trades": 0},
                "recent_trades": [],
            }

        return trading_bot_agent.get_performance_report()
    except Exception as e:
        system_logger.error(
            f"Error getting performance report: {str(e)}",
            additional_info={
                "context": "Get Performance Report",
            },
            exc_info=True,
        )
        return {"error": str(e)}


@app.get("/api/v1/trading/market/{symbol}")
async def get_market_data(symbol: str, token: str = Depends(verify_token)):
    """Get market data for a symbol"""
    try:
        if not trading_bot_agent:
            return {
                "symbol": symbol,
                "current_price": 150.0,
                "change_24h": 2.5,
                "volume": 1000000,
                "market_sentiment": "neutral",
            }

        market_conditions = trading_bot_agent.get_market_data(symbol)
        return {
            "symbol": market_conditions.symbol,
            "current_price": market_conditions.current_price,
            "volume": market_conditions.volume,
            "change_24h": market_conditions.change_24h,
            "rsi": market_conditions.rsi,
            "macd": market_conditions.macd,
            "moving_avg_20": market_conditions.moving_avg_20,
            "moving_avg_50": market_conditions.moving_avg_50,
            "volatility": market_conditions.volatility,
            "market_sentiment": market_conditions.market_sentiment,
        }
    except Exception as e:
        system_logger.error(
            f"Error getting market data: {str(e)}",
            additional_info={
                "context": "Get Market Data",
                "symbol": symbol,
            },
            exc_info=True,
        )
        print(f"Error getting market data: {e}")
        return {"error": str(e)}


# Database endpoints
@app.get("/api/v1/customers")
async def list_customers(
    limit: int = 10, offset: int = 0, token: str = Depends(verify_token)
):
    """List customers"""
    return {
        "customers": [
            {"id": i, "name": f"Customer {i}", "email": f"customer{i}@example.com"}
            for i in range(1, min(limit + 1, 6))
        ],
        "total": 5,
    }


@app.get("/api/v1/campaigns")
async def list_campaigns(
    limit: int = 10, offset: int = 0, token: str = Depends(verify_token)
):
    """List marketing campaigns"""
    return {
        "campaigns": [
            {
                "id": i,
                "name": f"Campaign {i}",
                "status": "active" if i % 2 == 0 else "completed",
            }
            for i in range(1, min(limit + 1, 4))
        ],
        "total": 3,
    }


@app.get("/api/v1/knowledge-base/search")
async def search_knowledge(
    query: str, limit: int = 10, token: str = Depends(verify_token)
):
    """Search knowledge base"""
    # Return mock results
    return {
        "results": [
            {
                "id": f"kb_{i}",
                "title": f"Knowledge Article {i}",
                "content": f"Content related to {query}",
            }
            for i in range(1, min(limit + 1, 3))
        ]
    }


# Debug endpoint
@app.get("/debug")
async def debug_info():
    """Debug information"""
    return {
        "config_loaded": bool(api_config),
        "agents_loaded": {
            "customer_support": customer_support_agent is not None,
            "marketing": marketing_agent is not None,
            "personal_virtual_assistant": pva_agent is not None,
            "financial_trading_bot": trading_bot_agent is not None,
        },
        "routes": [getattr(route, "path", str(route)) for route in app.routes],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=api_config["api"]["host"],
        port=api_config["api"]["port"],
        reload=api_config["api"]["reload"],
    )
