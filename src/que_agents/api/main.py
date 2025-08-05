# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module implements the main API for the Agentic AI system

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from que_agents.agents.personal_assistant_agent import (
    PersonalAssistantAgent,
)  # New import
from src.que_agents.agents.customer_support_agent import CustomerSupportAgent
from src.que_agents.agents.marketing_agent import (
    CampaignRequest,
    CampaignType,
    ContentType,
    MarketingAgent,
)
from src.que_agents.core.database import Customer, MarketingCampaign, get_session
from src.que_agents.knowledge_base.kb_manager import search_knowledge_base

# Load API configuration
with open("configs/api_config.yaml", "r") as f:
    api_config = yaml.safe_load(f)

# Load API configuration - fix the path
config_path = os.path.join(
    os.path.dirname(__file__), "../../../configs/api_config.yaml"
)
if not os.path.exists(config_path):
    # Alternative path
    config_path = "configs/api_config.yaml"

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


# Authentication dependency (simplified for demo)
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)"""
    # In production, implement proper JWT validation
    if credentials.credentials != api_config["authentication"]["api_token"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# Initialize agents
customer_support_agent = CustomerSupportAgent()
marketing_agent = MarketingAgent()
personal_assistant_agent = PersonalAssistantAgent()  # New agent initialization

# Pydantic models for request/response validation


class CustomerSupportRequest(BaseModel):
    customer_id: int = Field(..., description="Customer ID")
    message: str = Field(..., description="Customer message")


class CustomerSupportResponse(BaseModel):
    response: str
    confidence: float
    escalate: bool
    suggested_actions: List[str]
    knowledge_sources: List[str]
    sentiment: str
    timestamp: str


class MarketingCampaignRequest(BaseModel):
    campaign_type: str = Field(..., description="Type of campaign")
    target_audience: str = Field(..., description="Target audience description")
    budget: float = Field(..., gt=0, description="Campaign budget")
    duration_days: int = Field(..., gt=0, description="Campaign duration in days")
    goals: List[str] = Field(..., description="Campaign goals")
    channels: List[str] = Field(..., description="Marketing channels")
    content_requirements: List[str] = Field(..., description="Content types needed")


class ContentGenerationRequest(BaseModel):
    platform: str = Field(..., description="Social media platform")
    content_type: str = Field(..., description="Type of content")
    campaign_theme: str = Field(..., description="Campaign theme")
    target_audience: str = Field(..., description="Target audience")
    key_messages: List[str] = Field(..., description="Key messages to include")


class CampaignAnalysisRequest(BaseModel):
    campaign_id: int = Field(..., description="Campaign ID to analyze")


# New Pydantic models for Personal Assistant
class PersonalAssistantRequest(BaseModel):
    user_id: int = Field(..., description="User ID for the personal assistant")
    message: str = Field(..., description="User's request or command")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Previous chat history for context"
    )


class PersonalAssistantResponse(BaseModel):
    response: str
    tool_used: Optional[str]
    tool_output: Optional[Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agents: Dict[str, str]


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agents={
            "customer_support": "active",
            "marketing": "active",
            "personal_assistant": "active",  # New agent status
        },
    )


# Customer Support Agent endpoints
@app.post("/api/v1/customer-support/chat", response_model=CustomerSupportResponse)
async def customer_support_chat(
    request: CustomerSupportRequest, token: str = Depends(verify_token)
):
    """Handle customer support chat request"""
    try:
        result = customer_support_agent.handle_customer_request(
            customer_id=request.customer_id, message=request.message
        )
        return CustomerSupportResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing customer support request: {str(e)}",
        )


@app.get("/api/v1/customer-support/customer/{customer_id}")
async def get_customer_context(customer_id: int, token: str = Depends(verify_token)):
    """Get customer context information"""
    try:
        context = customer_support_agent.get_customer_context(customer_id)
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found"
            )
        return {
            "customer_id": context.customer_id,
            "name": context.name,
            "email": context.email,
            "tier": context.tier,
            "company": context.company,
            "recent_interactions": context.recent_interactions,
            "open_tickets": context.open_tickets,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving customer context: {str(e)}",
        )


# Marketing Agent endpoints
@app.post("/api/v1/marketing/campaign/create")
async def create_marketing_campaign(
    request: MarketingCampaignRequest, token: str = Depends(verify_token)
):
    """Create a new marketing campaign"""
    try:
        # Convert request to internal format
        campaign_request = CampaignRequest(
            campaign_type=CampaignType(request.campaign_type),
            target_audience=request.target_audience,
            budget=request.budget,
            duration_days=request.duration_days,
            goals=request.goals,
            channels=request.channels,
            content_requirements=[
                ContentType(ct) for ct in request.content_requirements
            ],
        )

        # Create campaign plan
        campaign_plan = marketing_agent.create_campaign_plan(campaign_request)

        return {
            "campaign_id": campaign_plan.campaign_id,
            "strategy": campaign_plan.strategy,
            "content_pieces": [
                {
                    "content_type": cp.content_type.value,
                    "platform": cp.platform,
                    "title": cp.title,
                    "content": cp.content,
                    "hashtags": cp.hashtags,
                    "call_to_action": cp.call_to_action,
                    "estimated_reach": cp.estimated_reach,
                }
                for cp in campaign_plan.content_pieces
            ],
            "schedule": campaign_plan.schedule,
            "budget_allocation": campaign_plan.budget_allocation,
            "success_metrics": campaign_plan.success_metrics,
            "estimated_performance": campaign_plan.estimated_performance,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid campaign type or content type: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating marketing campaign: {str(e)}",
        )


@app.post("/api/v1/marketing/content/generate")
async def generate_marketing_content(
    request: ContentGenerationRequest, token: str = Depends(verify_token)
):
    """Generate marketing content for specific platform"""
    try:
        content = marketing_agent.generate_content(
            platform=request.platform,
            content_type=ContentType(request.content_type),
            campaign_theme=request.campaign_theme,
            target_audience=request.target_audience,
            key_messages=request.key_messages,
        )

        return {
            "content_type": content.content_type.value,
            "platform": content.platform,
            "title": content.title,
            "content": content.content,
            "hashtags": content.hashtags,
            "call_to_action": content.call_to_action,
            "estimated_reach": content.estimated_reach,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(e)}",
        )


@app.post("/api/v1/marketing/campaign/analyze")
async def analyze_campaign_performance(
    request: CampaignAnalysisRequest, token: str = Depends(verify_token)
):
    """Analyze campaign performance and provide recommendations"""
    try:
        analysis = marketing_agent.analyze_campaign_performance(request.campaign_id)
        optimization = marketing_agent.optimize_campaign(request.campaign_id)

        return {
            "campaign_id": request.campaign_id,
            "analysis": analysis,
            "optimization": optimization,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing campaign: {str(e)}",
        )


# Personal Assistant Agent endpoints
@app.post("/api/v1/assistant/request", response_model=PersonalAssistantResponse)
async def personal_assistant_request(
    request: PersonalAssistantRequest, token: str = Depends(verify_token)
):
    """Handle personal assistant requests"""
    try:
        result = personal_assistant_agent.process_request(
            user_id=request.user_id,
            message=request.message,
            chat_history=(
                request.chat_history if request.chat_history is not None else []
            ),
        )
        return PersonalAssistantResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing personal assistant request: {str(e)}",
        )


# Knowledge base endpoints
@app.get("/api/v1/knowledge-base/search")
async def search_knowledge_base_api(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
    token: str = Depends(verify_token),
):
    """Search the knowledge base"""
    try:
        results = search_knowledge_base(
            query, category if category is not None else "", limit
        )
        return {"query": query, "category": category, "results": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching knowledge base: {str(e)}",
        )


# Database endpoints
@app.get("/api/v1/customers")
async def list_customers(
    limit: int = 10, offset: int = 0, token: str = Depends(verify_token)
):
    """List customers"""
    try:
        session = get_session()
        try:
            customers = session.query(Customer).offset(offset).limit(limit).all()
            return {
                "customers": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "email": c.email,
                        "tier": c.tier,
                        "company": c.company,
                        "created_at": (
                            c.created_at.isoformat()
                            if c.created_at is not None
                            else None
                        ),
                    }
                    for c in customers
                ],
                "total": session.query(Customer).count(),
            }
        finally:
            session.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing customers: {str(e)}",
        )


@app.get("/api/v1/campaigns")
async def list_campaigns(
    limit: int = 10, offset: int = 0, token: str = Depends(verify_token)
):
    """List marketing campaigns"""
    try:
        session = get_session()
        try:
            campaigns = (
                session.query(MarketingCampaign).offset(offset).limit(limit).all()
            )
            return {
                "campaigns": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "campaign_type": c.campaign_type,
                        "target_audience": c.target_audience,
                        "budget": float(getattr(c, "budget", 0.0)),
                        "status": c.status,
                        "start_date": (
                            c.start_date.isoformat()
                            if c.start_date is not None
                            else None
                        ),
                        "end_date": (
                            c.end_date.isoformat() if c.end_date is not None else None
                        ),
                    }
                    for c in campaigns
                ],
                "total": session.query(MarketingCampaign).count(),
            }
        finally:
            session.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing campaigns: {str(e)}",
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=api_config["api"]["host"],
        port=api_config["api"]["port"],
        reload=api_config["api"]["reload"],
        log_level=api_config["api"]["log_level"],
    )
